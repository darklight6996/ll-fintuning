"""
core/ingestion.py

Purpose
-------
Extensible multi-format file ingestion pipeline for common security scanner outputs,
cloud CLI configurations, and unstructured technical artifacts.

Responsibilities
----------------
1. Parse Nmap XML/text scans (hosts, open ports, protocols, services, versions).
2. Parse Burp Suite XML/text exports (endpoints, issue types, severities, descriptions).
3. Parse Cloud CLI JSON outputs (Azure/AWS IAM, Key Vaults, S3, policies).
4. Parse generic text files, logs, and configurations with heuristic entity extraction.
5. Provide structured normalization so extracted findings directly drive the dynamic planner.
"""

import os
import re
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class IngestionResult:
    """
    Standardized container for parsed artifact data.
    """

    def __init__(
        self,
        parser_type: str,
        source_path: str,
        summary: str,
        entities: Dict[str, List[Any]],
        details: Dict[str, Any],
        saved_artifact_path: Optional[str] = None
    ):
        self.parser_type = parser_type
        self.source_path = source_path
        self.summary = summary
        self.entities = entities
        self.details = details
        self.saved_artifact_path = saved_artifact_path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parser_type": self.parser_type,
            "source_path": self.source_path,
            "summary": self.summary,
            "entities": self.entities,
            "details": self.details,
            "saved_artifact_path": self.saved_artifact_path
        }


class BaseParser:
    """Base interface for security artifact parsers."""

    name: str = "base"

    def can_parse(self, file_path: Path, sample: str) -> bool:
        raise NotImplementedError

    def parse(self, file_path: Path) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        """
        Parse file.
        Returns:
            Tuple of (summary_str, entities_dict, details_dict)
        """
        raise NotImplementedError


class NmapParser(BaseParser):
    """Parses Nmap XML output files (and falls back to standard text output)."""

    name: str = "nmap"

    def can_parse(self, file_path: Path, sample: str) -> bool:
        lower_sample = sample[:2000].lower()
        if file_path.suffix.lower() == ".xml" and ("<nmaprun" in lower_sample or "nmap" in lower_sample):
            return True
        if "starting nmap" in lower_sample or "nmap scan report" in lower_sample:
            return True
        return False

    def parse(self, file_path: Path) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return self._parse_xml(root)
        except ET.ParseError:
            # Fallback to text parsing
            return self._parse_text(file_path)

    def _parse_xml(self, root: ET.Element) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        hosts_data = []
        all_ips = set()
        all_ports = set()
        all_domains = set()
        all_services = set()

        for host in root.findall("host"):
            # Check host status
            status = host.find("status")
            if status is not None and status.get("state") != "up":
                continue

            host_ips = []
            for addr in host.findall("address"):
                ip = addr.get("addr")
                if ip:
                    host_ips.append(ip)
                    all_ips.add(ip)

            hostnames = []
            hostnames_elem = host.find("hostnames")
            if hostnames_elem is not None:
                for hn in hostnames_elem.findall("hostname"):
                    name = hn.get("name")
                    if name:
                        hostnames.append(name)
                        all_domains.add(name)

            ports_elem = host.find("ports")
            ports_list = []
            if ports_elem is not None:
                for port in ports_elem.findall("port"):
                    port_id = port.get("portid")
                    protocol = port.get("protocol", "tcp")
                    state_elem = port.find("state")
                    state = state_elem.get("state") if state_elem is not None else "unknown"

                    if state == "open":
                        service_elem = port.find("service")
                        service_name = service_elem.get("name", "unknown") if service_elem is not None else "unknown"
                        product = service_elem.get("product", "") if service_elem is not None else ""
                        version = service_elem.get("version", "") if service_elem is not None else ""

                        try:
                            port_num = int(port_id)
                            all_ports.add(port_num)
                        except (ValueError, TypeError):
                            port_num = port_id

                        all_services.add(f"{service_name}:{port_id}")

                        scripts = []
                        for s in port.findall("script"):
                            scripts.append({"id": s.get("id"), "output": s.get("output", "")[:200]})

                        ports_list.append({
                            "port": port_num,
                            "protocol": protocol,
                            "state": state,
                            "service": service_name,
                            "product": product,
                            "version": version,
                            "scripts": scripts
                        })

            hosts_data.append({
                "ips": host_ips,
                "hostnames": hostnames,
                "open_ports": ports_list
            })

        summary = (
            f"Nmap scan with {len(hosts_data)} live host(s), "
            f"{len(all_ports)} unique open port(s) ({', '.join(str(p) for p in sorted(list(all_ports))[:10])}), "
            f"and {len(all_services)} identified service(s)."
        )

        entities = {
            "ips": sorted(list(all_ips)),
            "ports": sorted([p for p in all_ports if isinstance(p, int)]),
            "domains": sorted(list(all_domains)),
            "services": sorted(list(all_services)),
            "cloud_resources": [],
            "vulnerabilities": []
        }

        details = {"hosts": hosts_data}
        return summary, entities, details

    def _parse_text(self, file_path: Path) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        all_ips = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", content))
        all_ports = set()
        port_matches = re.findall(r"(\d+)/(tcp|udp)\s+open\s+([\w\-]+)", content)

        parsed_ports = []
        for p, proto, srv in port_matches:
            all_ports.add(int(p))
            parsed_ports.append({"port": int(p), "protocol": proto, "service": srv})

        summary = f"Text Nmap scan identified {len(all_ips)} IP(s) and {len(all_ports)} open port(s)."
        entities = {
            "ips": sorted(list(all_ips)),
            "ports": sorted(list(all_ports)),
            "domains": [],
            "services": [f"{item['service']}:{item['port']}" for item in parsed_ports],
            "cloud_resources": [],
            "vulnerabilities": []
        }
        details = {"raw_ports": parsed_ports}
        return summary, entities, details


class BurpParser(BaseParser):
    """Parses Burp Suite XML issue exports and HTTP item collections."""

    name: str = "burp"

    def can_parse(self, file_path: Path, sample: str) -> bool:
        lower_sample = sample[:2000].lower()
        if "<issues" in lower_sample or "<issue>" in lower_sample or "burpscan" in lower_sample:
            return True
        if "<items" in lower_sample and ("<url>" in lower_sample or "<host" in lower_sample):
            return True
        return False

    def parse(self, file_path: Path) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except Exception:
            return self._parse_generic_burp_text(file_path)

        issues = []
        all_hosts = set()
        all_vulns = set()

        for issue_elem in root.findall(".//issue"):
            name = issue_elem.findtext("name", "Unknown Finding")
            # NOTE: <host> element contains the host as text AND may have an 'ip' attribute.
            host_elem = issue_elem.find("host")
            host = host_elem.text.strip() if host_elem is not None and host_elem.text else ""
            ip = host_elem.get("ip", "").strip() if host_elem is not None else ""
            path = issue_elem.findtext("path", "")
            location = issue_elem.findtext("location", "")
            severity = issue_elem.findtext("severity", "Information")
            confidence = issue_elem.findtext("confidence", "")
            detail = issue_elem.findtext("issueDetail", "")[:300] if issue_elem.findtext("issueDetail") else ""

            if host:
                clean_host = re.sub(r"^https?://", "", host).split("/")[0].split(":")[0].strip()
                if clean_host:
                    all_hosts.add(clean_host)
            if ip and re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip.strip()):
                all_hosts.add(ip.strip())

            all_vulns.add(f"[{severity.upper()}] {name}")

            issues.append({
                "name": name,
                "host": host,
                "path": path,
                "location": location,
                "severity": severity,
                "confidence": confidence,
                "detail": detail
            })

        summary = f"Burp Suite report contains {len(issues)} finding(s) across {len(all_hosts)} host(s)."
        entities = {
            "ips": sorted([h for h in all_hosts if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", h)]),
            "ports": [80, 443],
            "domains": sorted([h for h in all_hosts if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", h)]),
            "services": ["http", "https"],
            "cloud_resources": [],
            "vulnerabilities": sorted(list(all_vulns))
        }
        details = {"issues": issues}
        return summary, entities, details

    def _parse_generic_burp_text(self, file_path: Path) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        urls = re.findall(r"https?://[^\s\"'>]+", content)
        summary = f"Burp export containing {len(urls)} target URL endpoints."
        entities = {
            "ips": [],
            "ports": [80, 443],
            "domains": list(set(re.findall(r"https?://([^/:\s]+)", content))),
            "services": ["http", "https"],
            "cloud_resources": [],
            "vulnerabilities": []
        }
        details = {"urls": urls[:50]}
        return summary, entities, details


class CloudCLIParser(BaseParser):
    """Parses JSON output from Azure CLI, AWS CLI, or GCP CLI."""

    name: str = "cloud"

    def can_parse(self, file_path: Path, sample: str) -> bool:
        if file_path.suffix.lower() != ".json":
            return False
        lower_sample = sample[:3000].lower()
        cloud_keywords = [
            "roledefinitionname", "principalid", "keyvault", "resourcename",
            "subscriptionid", "tenantid", "aws:iam", "policyname", "statement",
            "roleassignment", "arn:aws:", "s3::: ", "securitygroup", "instanceid"
        ]
        return any(k in lower_sample for k in cloud_keywords)

    def parse(self, file_path: Path) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            # Gracefully fall back to an empty result rather than crashing
            summary = f"Cloud CLI JSON parse error: {exc}. File may be malformed or truncated."
            entities: Dict[str, List[Any]] = {
                "ips": [], "ports": [], "domains": [],
                "services": [], "cloud_resources": [], "vulnerabilities": []
            }
            return summary, entities, {"error": str(exc)}

        cloud_resources = set()
        roles_discovered = set()
        principals = set()
        vulnerabilities = set()

        items = data if isinstance(data, list) else [data]

        for item in items:
            if not isinstance(item, dict):
                continue

            # Azure Role Assignment detection
            if "roleDefinitionName" in item or "roleDefinitionId" in item:
                role = item.get("roleDefinitionName", "Unknown Role")
                scope = item.get("scope", "")
                principal = item.get("principalName", item.get("principalId", ""))
                roles_discovered.add(role)
                if principal:
                    principals.add(principal)
                cloud_resources.add(f"RoleAssignment:{role}")
                if "Owner" in role or "Contributor" in role or "Administrator" in role:
                    vulnerabilities.add(f"High-Privilege Assignment: {role} ({principal})")

            # Key Vault detection
            if "vaultUri" in item or item.get("type", "").lower().endswith("/vaults"):
                name = item.get("name", "keyvault")
                cloud_resources.add(f"KeyVault:{name}")

            # AWS IAM Policy Statement detection
            if "Statement" in item and isinstance(item["Statement"], list):
                for stmt in item["Statement"]:
                    effect = stmt.get("Effect", "")
                    action = stmt.get("Action", "")
                    resource = stmt.get("Resource", "")
                    if effect == "Allow" and action in ("*", ["*"]) and resource in ("*", ["*"]):
                        vulnerabilities.add("AWS IAM Policy: Full Administrator Access (*:*)")
                    cloud_resources.add("AWS:IAMPolicy")

            # Storage / S3 detection
            if "BucketName" in item or "Buckets" in item:
                cloud_resources.add("AWS:S3Bucket")

        summary = (
            f"Cloud CLI output with {len(items)} record(s), {len(cloud_resources)} cloud resource(s), "
            f"and {len(roles_discovered)} role definition(s)."
        )

        entities = {
            "ips": [],
            "ports": [443],
            "domains": [],
            "services": ["cloud_api"],
            "cloud_resources": sorted(list(cloud_resources)),
            "vulnerabilities": sorted(list(vulnerabilities))
        }

        details = {
            "records_count": len(items),
            "roles": sorted(list(roles_discovered)),
            "principals": sorted(list(principals))
        }
        return summary, entities, details


class GenericTextParser(BaseParser):
    """Fallback parser for raw logs, configurations, scripts, and plain text notes."""

    name: str = "generic"

    def can_parse(self, file_path: Path, sample: str) -> bool:
        return True  # Fallback handles all readable text files

    def parse(self, file_path: Path) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
        content = file_path.read_text(encoding="utf-8", errors="replace")

        # Entity regex discovery
        ips = list(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", content)))
        domains = list(set(re.findall(r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\b", content)))
        domains = [d for d in domains if not d.replace(".", "").isdigit()]

        # Common sensitive security keywords
        sensitive_matches = []
        for pattern in [r"(?i)password\s*[:=]\s*(\S+)", r"(?i)bearer\s+([A-Za-z0-9\-\._~\+\/]+=*)", r"(?i)api[_-]?key\s*[:=]\s*(\S+)"]:
            found = re.findall(pattern, content)
            if found:
                sensitive_matches.extend(found[:3])

        vulnerabilities = []
        if sensitive_matches:
            vulnerabilities.append("Discovered hardcoded credential/token in file content")

        lines = content.splitlines()
        summary = f"Text artifact ({len(lines)} lines, {len(content)} bytes), identified {len(ips)} IP(s) and {len(domains)} domain(s)."

        entities = {
            "ips": sorted(ips),
            "ports": [],
            "domains": sorted(domains),
            "services": [],
            "cloud_resources": [],
            "vulnerabilities": vulnerabilities
        }

        preview = "\n".join(lines[:20])
        details = {
            "line_count": len(lines),
            "character_count": len(content),
            "preview": preview
        }
        return summary, entities, details


class FileIngestionEngine:
    """
    Orchestrates file inspection, parser selection, and entity normalization.
    """

    def __init__(self):
        # Ordered parsers (specific to generic)
        self.parsers: List[BaseParser] = [
            NmapParser(),
            BurpParser(),
            CloudCLIParser(),
            GenericTextParser()
        ]

    def ingest_file(
        self,
        file_path_str: str,
        artifacts_dir: Optional[Path] = None
    ) -> IngestionResult:
        """
        Inspect and parse a local file into structured findings.

        Args:
            file_path_str: String path to the target file.
            artifacts_dir: Optional destination to copy the raw artifact for evidence retention.

        Returns:
            IngestionResult object.
        """
        file_path = Path(file_path_str).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Target ingestion file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")

        # Read sample for header detection
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(4096)
        except Exception as e:
            raise IOError(f"Could not read file {file_path}: {e}")

        # Select parser
        selected_parser: Optional[BaseParser] = None
        for parser in self.parsers:
            if parser.can_parse(file_path, sample):
                selected_parser = parser
                break

        if not selected_parser:
            selected_parser = GenericTextParser()

        summary, entities, details = selected_parser.parse(file_path)

        # Retain raw copy in session artifacts if destination provided
        saved_path = None
        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            destination = artifacts_dir / file_path.name
            try:
                shutil.copy2(file_path, destination)
                saved_path = str(destination)
            except Exception as copy_err:
                # Non-fatal: log reason but continue with original path
                saved_path = str(file_path)
                import warnings
                warnings.warn(f"Could not copy artifact to session dir: {copy_err}")

        return IngestionResult(
            parser_type=selected_parser.name,
            source_path=str(file_path),
            summary=summary,
            entities=entities,
            details=details,
            saved_artifact_path=saved_path
        )

    def analyze_terminal_text(self, content: str) -> Dict[str, Any]:
        """
        Analyze pasted terminal output (e.g. Nmap scan, CrackMapExec/NetExec output,
        or tool logs) to discover ports, services, hostnames, and environment profiles.
        """
        import re

        # Extract IPs
        ips = list(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", content)))

        # Extract Ports & Services
        port_pattern = r"(\d{1,5})/(tcp|udp)\s+open\s+([\w\-]+)(?:\s+([^\r\n]+))?"
        matches = re.findall(port_pattern, content)

        friendly_service_map = {
            53: "DNS",
            88: "Kerberos",
            135: "RPC",
            139: "NetBIOS",
            389: "LDAP",
            445: "SMB",
            464: "kpasswd",
            593: "RPC-over-HTTP",
            636: "LDAPS",
            3268: "Global Catalog (LDAP)",
            3269: "Global Catalog SSL",
            80: "HTTP (Web)",
            443: "HTTPS (Web)",
            8080: "HTTP-Proxy",
            8443: "HTTPS-Alt",
            22: "SSH",
            21: "FTP",
            23: "Telnet",
            25: "SMTP",
            110: "POP3",
            143: "IMAP",
            3389: "RDP",
            1433: "MSSQL",
            3306: "MySQL",
            5432: "PostgreSQL",
            5985: "WinRM",
            5986: "WinRM (HTTPS)"
        }

        ports_found = set()
        detected_services = []
        raw_services = []

        for p_str, proto, srv_name, version_info in matches:
            p_num = int(p_str)
            ports_found.add(p_num)
            raw_services.append(f"{srv_name}:{p_num}")
            friendly = friendly_service_map.get(p_num, srv_name.upper())
            if friendly not in detected_services:
                detected_services.append(friendly)

        # Hostname / OS indicators
        host_match = re.search(r"(?i)(?:host|netbios\s+name|computer\s+name):\s*([A-Za-z0-9_\-]+)", content)
        host_name = host_match.group(1) if host_match else None

        os_match = re.search(r"(?i)(?:os|operating\s+system):\s*([A-Za-z0-9_\- ]+)", content)
        os_info = os_match.group(1).strip() if os_match else None

        domain_match = re.search(r"(?i)(?:domain|realm):\s*([A-Za-z0-9_\-\.]+)", content)
        domain_name = domain_match.group(1) if domain_match else None

        # Web & Bug Bounty Specific Detection (ffuf, httpx, subfinder, nuclei, burp)
        web_endpoints = re.findall(r"(?:200|301|302|403|401)\s+[-–—\s\d\w]+\s+(/\S+)", content)
        if not web_endpoints:
            web_endpoints = re.findall(r"(?:URL|Target|Path|Endpoint):\s*(https?://\S+|/\S+)", content, re.IGNORECASE)

        subdomains = [d for d in re.findall(r"\b([a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z]{2,})\b", content) if not d.replace(".", "").isdigit()]

        vuln_matches = re.findall(r"\[(info|low|medium|high|critical)\]\s+\[([\w\-]+)\]", content, re.IGNORECASE)
        found_vulns = [f"[{sev.upper()}] {vname}" for sev, vname in vuln_matches]

        # Environment Profile Classification
        lower_content = content.lower()
        has_kerberos = 88 in ports_found or "kerberos" in lower_content
        has_ldap = 389 in ports_found or 636 in ports_found or 3268 in ports_found or "ldap" in lower_content
        has_smb = 445 in ports_found or 139 in ports_found or "microsoft-ds" in lower_content
        has_dns = 53 in ports_found
        has_web = any(p in ports_found for p in (80, 443, 8080, 8443, 3000, 5000)) or "http" in lower_content or web_endpoints or subdomains

        if (has_kerberos or has_ldap) and (has_smb or has_dns):
            profile = "Likely Active Directory."
            suggested = "SMB enumeration."
        elif has_smb or 135 in ports_found or 3389 in ports_found or "windows" in lower_content:
            profile = "Windows Infrastructure Host."
            suggested = "SMB share enumeration."
        elif len(subdomains) > 3 or "subfinder" in lower_content or "httpx" in lower_content:
            profile = "Multi-Domain Web Attack Surface (Bug Bounty Target)."
            suggested = "HTTP service probing and technology profiling."
        elif web_endpoints or has_web:
            profile = "Web Application / API Surface."
            suggested = "Web directory fuzzing and parameter discovery."
        elif 22 in ports_found:
            profile = "Linux Host."
            suggested = "SSH and local service enumeration."
        else:
            profile = "Network Host."
            suggested = "Service enumeration."

        if web_endpoints and not detected_services:
            detected_services.append("HTTP Endpoints")
        if subdomains and "DNS Subdomains" not in detected_services:
            detected_services.append(f"{len(subdomains)} Subdomain(s)")

        all_domains = list(set(([domain_name] if domain_name else []) + subdomains[:10]))

        entities = {
            "ips": ips,
            "ports": sorted(list(ports_found)),
            "domains": all_domains,
            "services": raw_services,
            "cloud_resources": [],
            "vulnerabilities": found_vulns
        }

        return {
            "detected_services": detected_services,
            "environment_profile": profile,
            "host_name": host_name,
            "os_info": os_info,
            "domain_name": domain_name or (subdomains[0] if subdomains else None),
            "web_endpoints": web_endpoints[:15],
            "suggested_next_step": suggested,
            "entities": entities,
            "summary": f"Detected {len(detected_services)} item(s): {', '.join(detected_services[:6])}. {profile}"
        }
