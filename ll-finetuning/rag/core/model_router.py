"""
core/model_router.py

Purpose:
This file is responsible for loading the language model (LLM) and tokenizer,
handling CPU vs GPU model loading settings, and exposing a single clean
generate() method that the rest of the RAG pipeline can call.

Why this matters:
- Keeps model-loading logic separate from retrieval logic.
- Lets CPU and GPU pipelines use the SAME interface.
- Makes it easy later to swap models without rewriting the whole pipeline.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch


class ModelRouter:
    """
    ModelRouter is the abstraction layer between your pipeline and the actual LLM.

    Instead of your CPU/GPU scripts directly doing:
        tokenizer = AutoTokenizer.from_pretrained(...)
        model = AutoModelForSeq2SeqLM.from_pretrained(...)
        ...
        outputs = model.generate(...)

    they will call ModelRouter, and ModelRouter handles all of that.

    Main responsibilities:
    1. Load the correct tokenizer
    2. Load the correct model
    3. Put the model on the right device (CPU or GPU)
    4. Expose a clean generate(prompt) function
    """

    def __init__(
        self,
        model_name: str,
        mode: str = "cpu",
        max_input_tokens: int = 2048,
        max_new_tokens: int = 256
    ):
        """
        Parameters
        ----------
        model_name : str
            Hugging Face model name or local model path.

            Example CPU model:
                "google/flan-t5-base"

            Example GPU model:
                "mistralai/Mistral-7B-Instruct-v0.2"

        mode : str
            Either "cpu" or "gpu".
            This controls how the model is loaded.

        max_input_tokens : int
            Maximum number of input tokens the prompt can use.
            If prompt is too long, tokenizer will truncate.

        max_new_tokens : int
            Maximum number of tokens the model is allowed to generate.
        """

        self.model_name = model_name
        self.mode = mode.lower()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

        self.tokenizer = None
        self.model = None
        self.device = "cpu"

        self._load_model()

    # ============================================================
    # INTERNAL MODEL LOADER
    # ============================================================

    def _load_model(self):
        """
        Loads tokenizer + model depending on CPU or GPU mode.

        Design choice:
        - CPU pipeline will likely use Flan-T5 (seq2seq model)
        - GPU pipeline may use Mistral / causal LM

        So this router supports BOTH:
        - Seq2Seq models via AutoModelForSeq2SeqLM
        - Causal models via AutoModelForCausalLM

        How do we decide?
        Simple rule:
        - if model name contains "flan" or "t5" -> treat as seq2seq
        - otherwise -> treat as causal LM
        """

        print(f"Loading tokenizer for: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Some causal models do not have a pad token set.
        # If pad_token is missing, use eos_token as fallback.
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"pad token is not set for {self.model_name}")

        # ------------------------------------------------------------
        # Decide model family
        # ------------------------------------------------------------
        lower_name = self.model_name.lower()

        if "t5" in lower_name or "flan" in lower_name:
            model_family = "seq2seq"
        else:
            model_family = "causal"

        print(f"Detected model family: {model_family}")

        # ------------------------------------------------------------
        # CPU MODE
        # ------------------------------------------------------------
        if self.mode == "cpu":
            self.device = "cpu"
            print("Loading model in CPU mode...")

            if model_family == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name
                )

            self.model.to(self.device)

        # ------------------------------------------------------------
        # GPU MODE
        # ------------------------------------------------------------
        elif self.mode == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested, but CUDA is not available.")

            self.device = "cuda"
            print("Loading model in GPU mode...")

            if model_family == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )

            # Important:
            # if device_map="auto" is used, Hugging Face may already place
            # the model on the GPU. Calling .to("cuda") again is not necessary.
            if model_family == "seq2seq":
                self.model.to(self.device)

        else:
            raise ValueError("mode must be either 'cpu' or 'gpu'")

        self.model.eval()
        print("Model loaded successfully.")

    # ============================================================
    # PROMPT TOKENIZATION
    # ============================================================

    def _prepare_inputs(self, prompt: str):
        """
        Tokenize the input prompt into tensors.

        Why this is separated into its own function:
        - keeps generate() cleaner
        - easier to debug prompt/tokenization issues
        - later you can add logging, token counts, prompt inspection, etc.

        Returns
        -------
        dict
            Tokenized tensors ready to be passed into model.generate()
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        )

        # Move tokenized tensors to correct device if needed
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    # ============================================================
    # GENERATION
    # ============================================================

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response from the loaded model.
        """

        inputs = self._prepare_inputs(prompt)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # For causal models, decode only the newly generated tokens
        if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            generated_tokens = outputs[0][input_length:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return decoded.strip()

    # ============================================================
    # OPTIONAL DEBUG METHOD
    # ============================================================

    def info(self) -> dict:
        """
        Return basic information about the loaded model.
        Useful for debugging and logging.
        """
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "device": self.device,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens
        }