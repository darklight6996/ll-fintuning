"""
core/variable_engine.py

Purpose
-------
Make YAML procedures reusable across environments by substituting variables.

Responsibilities
----------------
1. Maintain global and per-assessment variable values (e.g. TENANT, TARGET_IP, USER, DOMAIN).
2. Substitute placeholder tokens like [VARIABLE_NAME] or {VARIABLE_NAME} inside text and commands.
3. Identify unpopulated variables requiring user input.
"""

import re
from typing import Dict, List, Set, Any, Union, Optional


class VariableEngine:
    """
    Engine for replacing environment variables inside command strings and templates.
    """

    def __init__(self, initial_vars: Optional[Dict[str, str]] = None):
        """
        Initialize the variable engine with optional default variables.
        """
        self.variables: Dict[str, str] = {}
        if initial_vars:
            for k, v in initial_vars.items():
                self.set_variable(k, str(v))

    def set_variable(self, name: str, value: str):
        """
        Set or update a variable value.
        Normalizes variable name to upper case without brackets.
        """
        clean_name = name.strip(" []{}").upper()
        self.variables[clean_name] = str(value)

    def set_variables(self, var_dict: Dict[str, str]):
        """
        Bulk update variables.
        """
        for k, v in var_dict.items():
            self.set_variable(k, v)

    def get_variable(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieve a variable value by name.
        """
        clean_name = name.strip(" []{}").upper()
        return self.variables.get(clean_name, default)

    def extract_variables(self, text: str) -> List[str]:
        """
        Find all placeholders in format [VAR_NAME] or {VAR_NAME} in text.

        Returns:
            List of unique variable names found.
        """
        # Matches [VAR_NAME] or {VAR_NAME} where VAR_NAME consists of word chars
        bracket_pattern = r'\[([A-Za-z0-9_]+)\]'
        curly_pattern = r'\{([A-Za-z0-9_]+)\}'

        found: Set[str] = set()
        for match in re.finditer(bracket_pattern, text):
            found.add(match.group(1).upper())
        for match in re.finditer(curly_pattern, text):
            found.add(match.group(1).upper())

        return sorted(list(found))

    def substitute(self, content: Union[str, List[str], Dict[str, Any]], custom_vars: Optional[Dict[str, str]] = None) -> Any:
        """
        Recursively substitute variable placeholders in strings, lists, or dicts.

        Args:
            content: Target string or nested data structure
            custom_vars: One-off variable overrides for this substitution

        Returns:
            Substituted content in matching type.
        """
        active_vars = dict(self.variables)
        if custom_vars:
            for k, v in custom_vars.items():
                clean_k = k.strip(" []{}").upper()
                active_vars[clean_k] = str(v)

        if isinstance(content, str):
            result = content
            for var_name, var_val in active_vars.items():
                # Replace [VAR_NAME] and {VAR_NAME}
                result = result.replace(f"[{var_name}]", var_val)
                result = result.replace(f"{{{var_name}}}", var_val)
                # Also handle lowercase variant placeholders if present e.g. [target_org]
                result = result.replace(f"[{var_name.lower()}]", var_val)
                result = result.replace(f"{{{var_name.lower()}}}", var_val)
            return result

        elif isinstance(content, list):
            return [self.substitute(item, custom_vars=custom_vars) for item in content]

        elif isinstance(content, dict):
            return {k: self.substitute(v, custom_vars=custom_vars) for k, v in content.items()}

        return content

    def get_missing_variables(self, text: str) -> List[str]:
        """
        Return list of variable placeholders in text that do NOT have a set value.
        """
        extracted = self.extract_variables(text)
        missing = []
        for var in extracted:
            if var not in self.variables and var.lower() not in [v.lower() for v in self.variables]:
                missing.append(var)
        return missing
