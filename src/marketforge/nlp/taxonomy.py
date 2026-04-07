"""
MarketForge AI — Skill Taxonomy & NLP Extraction Pipeline.

Three-gate architecture:
  Gate 1 — flashtext exact match against canonical skill taxonomy   (~80%)
  Gate 2 — spaCy EntityRuler + NER patterns                         (~15%)
  Gate 3 — Gemini Flash on unresolved tokens only                   (~5%)

Salary NER:  regex-based extraction of £X, £Xk, £X–£Y from body text.
Role Classifier: title pattern matching → ML fallback.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from marketforge.models.job import RawJob, RoleCategory

logger = structlog.get_logger(__name__)

# ── Canonical skill taxonomy ───────────────────────────────────────────────────
# Each entry: {"canonical": "X", "aliases": [...], "category": "...", "parent": "..."}
# This is a representative seed set — the SkillExtractionModelAgent grows it over time.

SKILL_TAXONOMY: list[dict] = [
    # ── Deep Learning frameworks ──────────────────────────────────────────────
    {"canonical": "PyTorch",        "aliases": ["torch", "pytorch", "py-torch"],         "category": "dl_framework"},
    {"canonical": "TensorFlow",     "aliases": ["tensorflow", "tf", "keras"],             "category": "dl_framework"},
    {"canonical": "JAX",            "aliases": ["jax", "flax", "haiku"],                  "category": "dl_framework"},
    {"canonical": "ONNX",           "aliases": ["onnx", "onnx-runtime"],                  "category": "dl_framework"},
    {"canonical": "Triton",         "aliases": ["triton-inference", "nvidia-triton"],     "category": "dl_framework"},

    # ── LLM / Agentic ─────────────────────────────────────────────────────────
    {"canonical": "LangChain",      "aliases": ["langchain"],                              "category": "llm_framework"},
    {"canonical": "LangGraph",      "aliases": ["langgraph"],                              "category": "llm_framework"},
    {"canonical": "LlamaIndex",     "aliases": ["llama-index", "llamaindex"],             "category": "llm_framework"},
    {"canonical": "Hugging Face",   "aliases": ["huggingface", "hf", "transformers", "hugging-face"], "category": "llm_framework"},
    {"canonical": "OpenAI API",     "aliases": ["openai", "gpt-4", "gpt-3.5", "chatgpt"], "category": "llm_provider"},
    {"canonical": "Gemini",         "aliases": ["gemini", "google-gemini", "gemini-pro"], "category": "llm_provider"},
    {"canonical": "Anthropic Claude","aliases": ["claude", "anthropic"],                  "category": "llm_provider"},
    {"canonical": "Llama",          "aliases": ["llama-3", "llama-2", "meta-llama"],     "category": "llm_provider"},
    {"canonical": "Mistral",        "aliases": ["mistral", "mistral-ai"],                 "category": "llm_provider"},
    {"canonical": "RAG",            "aliases": ["retrieval augmented generation", "rag"], "category": "llm_technique"},
    {"canonical": "Fine-tuning",    "aliases": ["fine-tuning", "finetuning", "lora", "qlora", "peft"], "category": "llm_technique"},
    {"canonical": "RLHF",           "aliases": ["rlhf", "reinforcement learning from human feedback"], "category": "llm_technique"},
    {"canonical": "Prompt engineering", "aliases": ["prompt engineering", "few-shot", "chain-of-thought"], "category": "llm_technique"},
    {"canonical": "Vector database","aliases": ["vector db", "vector store", "embedding store"], "category": "llm_infra"},
    {"canonical": "ChromaDB",       "aliases": ["chromadb", "chroma"],                   "category": "vector_db"},
    {"canonical": "Pinecone",       "aliases": ["pinecone"],                              "category": "vector_db"},
    {"canonical": "Weaviate",       "aliases": ["weaviate"],                              "category": "vector_db"},
    {"canonical": "Qdrant",         "aliases": ["qdrant"],                                "category": "vector_db"},
    {"canonical": "FAISS",          "aliases": ["faiss", "facebook-ai-similarity-search"],"category": "vector_db"},
    {"canonical": "Milvus",         "aliases": ["milvus"],                                "category": "vector_db"},

    # ── MLOps ─────────────────────────────────────────────────────────────────
    {"canonical": "MLflow",         "aliases": ["mlflow"],                                "category": "mlops"},
    {"canonical": "Kubeflow",       "aliases": ["kubeflow"],                              "category": "mlops"},
    {"canonical": "Apache Airflow", "aliases": ["airflow", "apache airflow"],             "category": "mlops"},
    {"canonical": "Prefect",        "aliases": ["prefect"],                               "category": "mlops"},
    {"canonical": "Weights & Biases","aliases": ["wandb", "weights and biases", "w&b"],  "category": "mlops"},
    {"canonical": "DVC",            "aliases": ["dvc", "data version control"],           "category": "mlops"},
    {"canonical": "BentoML",        "aliases": ["bentoml"],                               "category": "mlops"},
    {"canonical": "Ray",            "aliases": ["ray", "ray-tune", "raytune", "ray-serve"],"category": "mlops"},
    {"canonical": "Seldon",         "aliases": ["seldon", "seldon-core"],                 "category": "mlops"},
    {"canonical": "Feature Store",  "aliases": ["feast", "tecton", "hopsworks"],          "category": "mlops"},
    {"canonical": "Model monitoring","aliases": ["model monitoring", "evidently", "arize", "whylabs"], "category": "mlops"},

    # ── Classic ML ────────────────────────────────────────────────────────────
    {"canonical": "scikit-learn",   "aliases": ["sklearn", "scikit learn", "scikit-learn"], "category": "ml_library"},
    {"canonical": "XGBoost",        "aliases": ["xgboost", "xgb"],                       "category": "ml_library"},
    {"canonical": "LightGBM",       "aliases": ["lightgbm", "lgbm"],                     "category": "ml_library"},
    {"canonical": "CatBoost",       "aliases": ["catboost"],                              "category": "ml_library"},
    {"canonical": "SHAP",           "aliases": ["shap", "shapley"],                       "category": "ml_library"},
    {"canonical": "Optuna",         "aliases": ["optuna"],                                "category": "ml_library"},
    {"canonical": "statsmodels",    "aliases": ["statsmodels"],                           "category": "ml_library"},
    {"canonical": "Prophet",        "aliases": ["prophet", "facebook prophet"],           "category": "ml_library"},

    # ── NLP specific ──────────────────────────────────────────────────────────
    {"canonical": "spaCy",          "aliases": ["spacy", "spaCy"],                        "category": "nlp"},
    {"canonical": "NLTK",           "aliases": ["nltk"],                                  "category": "nlp"},
    {"canonical": "BERT",           "aliases": ["bert", "roberta", "distilbert", "albert"], "category": "nlp"},
    {"canonical": "SBERT",          "aliases": ["sbert", "sentence-transformers", "sentence transformers"], "category": "nlp"},
    {"canonical": "T5",             "aliases": ["t5", "flan-t5"],                         "category": "nlp"},

    # ── Computer Vision ───────────────────────────────────────────────────────
    {"canonical": "OpenCV",         "aliases": ["opencv", "cv2"],                         "category": "computer_vision"},
    {"canonical": "YOLO",           "aliases": ["yolo", "yolov8", "yolov5", "ultralytics"],"category": "computer_vision"},
    {"canonical": "Detectron2",     "aliases": ["detectron2"],                            "category": "computer_vision"},
    {"canonical": "MMDetection",    "aliases": ["mmdetection"],                           "category": "computer_vision"},
    {"canonical": "Stable Diffusion","aliases": ["stable diffusion", "sdxl", "diffusers"],"category": "generative_ai"},
    {"canonical": "Segment Anything","aliases": ["sam", "segment anything"],              "category": "computer_vision"},

    # ── Data Engineering ──────────────────────────────────────────────────────
    {"canonical": "Apache Spark",   "aliases": ["spark", "pyspark", "apache spark"],     "category": "data_engineering"},
    {"canonical": "Apache Kafka",   "aliases": ["kafka", "apache kafka"],                 "category": "data_engineering"},
    {"canonical": "dbt",            "aliases": ["dbt", "dbt-core"],                       "category": "data_engineering"},
    {"canonical": "Databricks",     "aliases": ["databricks"],                            "category": "data_engineering"},
    {"canonical": "Snowflake",      "aliases": ["snowflake"],                             "category": "data_warehouse"},
    {"canonical": "BigQuery",       "aliases": ["bigquery", "google bigquery"],           "category": "data_warehouse"},
    {"canonical": "Redshift",       "aliases": ["redshift", "amazon redshift"],          "category": "data_warehouse"},
    {"canonical": "Airbyte",        "aliases": ["airbyte"],                               "category": "data_engineering"},
    {"canonical": "Fivetran",       "aliases": ["fivetran"],                              "category": "data_engineering"},
    {"canonical": "Great Expectations","aliases": ["great expectations", "gx"],          "category": "data_quality"},

    # ── Languages ─────────────────────────────────────────────────────────────
    {"canonical": "Python",         "aliases": ["python", "python3", "python 3"],         "category": "language"},
    {"canonical": "SQL",            "aliases": ["sql", "mysql", "postgresql", "postgres", "sqlite", "t-sql", "pl/sql"], "category": "language"},
    {"canonical": "R",              "aliases": ["r programming", "r language", "rstudio", "tidyverse", "ggplot2", "dplyr", "caret", "shiny"], "category": "language"},
    {"canonical": "Scala",          "aliases": ["scala"],                                 "category": "language"},
    {"canonical": "Java",           "aliases": ["java"],                                  "category": "language"},
    {"canonical": "Go",             "aliases": ["golang", "go programming", "go lang"],   "category": "language"},
    {"canonical": "Rust",           "aliases": ["rust"],                                  "category": "language"},
    {"canonical": "C++",            "aliases": ["c++", "cpp", "c plus plus"],             "category": "language"},
    {"canonical": "Julia",          "aliases": ["julia"],                                 "category": "language"},

    # ── Cloud ──────────────────────────────────────────────────────────────────
    {"canonical": "AWS",            "aliases": ["amazon web services", "aws", "ec2", "s3", "sagemaker", "bedrock"], "category": "cloud"},
    {"canonical": "GCP",            "aliases": ["google cloud", "gcp", "vertex ai", "gke"], "category": "cloud"},
    {"canonical": "Azure",          "aliases": ["microsoft azure", "azure", "azure ml"],  "category": "cloud"},
    {"canonical": "Kubernetes",     "aliases": ["kubernetes", "k8s"],                     "category": "infra"},
    {"canonical": "Docker",         "aliases": ["docker", "containers", "containerisation"], "category": "infra"},
    {"canonical": "Terraform",      "aliases": ["terraform", "iac"],                      "category": "infra"},

    # ── Backend / APIs ────────────────────────────────────────────────────────
    {"canonical": "FastAPI",        "aliases": ["fastapi"],                               "category": "backend"},
    {"canonical": "Flask",          "aliases": ["flask"],                                 "category": "backend"},
    {"canonical": "Django",         "aliases": ["django"],                                "category": "backend"},
    {"canonical": "REST API",       "aliases": ["rest api", "restful", "rest"],           "category": "backend"},
    {"canonical": "GraphQL",        "aliases": ["graphql"],                               "category": "backend"},
    {"canonical": "gRPC",           "aliases": ["grpc"],                                  "category": "backend"},
    {"canonical": "PostgreSQL",     "aliases": ["postgresql", "postgres"],                "category": "database"},
    {"canonical": "MongoDB",        "aliases": ["mongodb", "mongo"],                      "category": "database"},
    {"canonical": "Redis",          "aliases": ["redis"],                                 "category": "database"},
    {"canonical": "Elasticsearch",  "aliases": ["elasticsearch", "elastic", "opensearch"],"category": "database"},

    # ── General engineering ────────────────────────────────────────────────────
    {"canonical": "Git",            "aliases": ["git", "github", "gitlab"],               "category": "devops"},
    {"canonical": "CI/CD",          "aliases": ["ci/cd", "continuous integration", "github actions", "jenkins"], "category": "devops"},
    {"canonical": "Pytest",         "aliases": ["pytest", "unit testing", "test-driven development"], "category": "testing"},
    {"canonical": "Pandas",         "aliases": ["pandas"],                                "category": "data_analysis"},
    {"canonical": "NumPy",          "aliases": ["numpy", "np"],                           "category": "data_analysis"},
    {"canonical": "Polars",         "aliases": ["polars"],                                "category": "data_analysis"},
    {"canonical": "Matplotlib",     "aliases": ["matplotlib"],                            "category": "visualisation"},
    {"canonical": "Plotly",         "aliases": ["plotly"],                                "category": "visualisation"},

    # ── Domain concepts ────────────────────────────────────────────────────────
    {"canonical": "Multi-agent systems","aliases": ["multi-agent", "agentic ai", "agent frameworks"], "category": "ai_concept"},
    {"canonical": "Reinforcement learning","aliases": ["reinforcement learning", "rl", "deep rl", "ppo", "dqn"], "category": "ai_concept"},
    {"canonical": "Federated learning","aliases": ["federated learning"],                "category": "ai_concept"},
    {"canonical": "AI safety",      "aliases": ["ai safety", "alignment", "ai alignment", "red teaming"], "category": "ai_concept"},
    {"canonical": "MLOps",          "aliases": ["mlops", "ml operations"],                "category": "ai_concept"},
    {"canonical": "A/B testing",    "aliases": ["a/b testing", "ab testing", "experimentation"], "category": "data_science"},
    {"canonical": "Causal inference","aliases": ["causal inference", "causality"],        "category": "data_science"},
    {"canonical": "Time series",    "aliases": ["time series", "forecasting", "arima", "lstm"], "category": "data_science"},
    {"canonical": "NLP",            "aliases": ["natural language processing", "nlp", "text mining"], "category": "ai_domain"},
    {"canonical": "Computer Vision","aliases": ["computer vision", "image recognition", "object detection"], "category": "ai_domain"},
    {"canonical": "Speech recognition","aliases": ["speech recognition", "asr", "whisper"], "category": "ai_domain"},
]


class SkillTaxonomy:
    """
    In-memory taxonomy loaded from SKILL_TAXONOMY.
    The FlashText processor enables O(1) multi-keyword matching.
    """

    def __init__(self) -> None:
        self._canonical_map: dict[str, str] = {}   # alias_lower → canonical
        self._category_map:  dict[str, str] = {}   # canonical  → category
        self._processor = None
        self._load()

    def _load(self) -> None:
        try:
            from flashtext import KeywordProcessor
            self._processor = KeywordProcessor(case_sensitive=False)
        except ImportError:
            logger.warning("flashtext.not_installed — falling back to regex matching")

        for entry in SKILL_TAXONOMY:
            canonical = entry["canonical"]
            category  = entry.get("category", "general")
            self._category_map[canonical] = category

            all_aliases = [canonical.lower()] + [a.lower() for a in entry.get("aliases", [])]
            for alias in all_aliases:
                self._canonical_map[alias.strip()] = canonical

            if self._processor is not None:
                # Only register canonical as a keyword if it's unambiguous (length > 2
                # and not a common English word). Single/short names like "R", "Go", "C"
                # must be matched via explicit aliases only to avoid false positives.
                _ambiguous = {"R", "C", "Go", "C#", "F#"}
                if canonical not in _ambiguous and len(canonical) > 2:
                    self._processor.add_keyword(canonical, canonical)
                for alias in entry.get("aliases", []):
                    self._processor.add_keyword(alias, canonical)

        logger.info("taxonomy.loaded", skills=len(SKILL_TAXONOMY), aliases=len(self._canonical_map))

    def extract(self, text: str) -> list[tuple[str, str]]:
        """
        Gate 1: exact taxonomy match.
        Returns list of (canonical_skill, category).
        """
        if not text:
            return []

        if self._processor is not None:
            found = self._processor.extract_keywords(text, span_info=False)
            return [(skill, self._category_map.get(skill, "general")) for skill in set(found)]

        # Fallback: simple substring matching
        text_lower = text.lower()
        results: list[tuple[str, str]] = []
        seen: set[str] = set()
        for alias, canonical in self._canonical_map.items():
            if alias in text_lower and canonical not in seen:
                seen.add(canonical)
                results.append((canonical, self._category_map.get(canonical, "general")))
        return results

    def resolve(self, term: str) -> str | None:
        """Resolve an alias to its canonical form."""
        return self._canonical_map.get(term.lower().strip())

    @property
    def all_canonical(self) -> list[str]:
        return list(self._category_map.keys())


# ── Gate 2: spaCy NER ────────────────────────────────────────────────────────
class SpacyGate:
    """
    Gate 2: spaCy EntityRuler + NER for skills not caught by exact matching.
    Uses context patterns like "experience with X", "proficiency in X".
    Loads lazily to avoid slow startup.
    """

    _nlp = None
    _loaded = False

    @classmethod
    def _load(cls) -> None:
        if cls._loaded:
            return
        try:
            import spacy
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except OSError:
                logger.warning("spacy.model.not_found — run: python -m spacy download en_core_web_sm")
                cls._loaded = True
                return
            # Add EntityRuler before NER for technology names
            ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
            patterns = [
                {"label": "TECH", "pattern": [{"LOWER": {"IN": list(_TECH_PATTERNS)}}]},
                # Captures "experience with X" / "knowledge of X" / "proficiency in X"
                {"label": "TECH", "pattern": [{"LOWER": {"IN": ["experience", "knowledge", "proficiency", "expertise"]}}, {"LOWER": {"IN": ["with", "in", "of"]}}, {"IS_ALPHA": True}]},
            ]
            ruler.add_patterns(patterns)
            cls._nlp = nlp
            cls._loaded = True
            logger.info("spacy.gate.ready")
        except Exception as exc:
            logger.warning("spacy.gate.load_failed", error=str(exc))
            cls._loaded = True

    def extract(self, text: str, already_found: set[str]) -> list[str]:
        """Returns raw token strings that spaCy classified as TECH/PRODUCT entities."""
        self._load()
        if self._nlp is None:
            return []
        try:
            doc = self._nlp(text[:5000])  # cap to 5000 chars for speed
            found = []
            for ent in doc.ents:
                if ent.label_ in ("TECH", "PRODUCT", "ORG"):
                    token = ent.text.strip()
                    if token not in already_found and len(token) > 2:
                        found.append(token)
            return found
        except Exception:
            return []


# ── Gate 3: LLM fallback ──────────────────────────────────────────────────────
class LLMGate:
    """
    Gate 3: Gemini Flash for unresolved tokens.
    Called only when Gates 1+2 leave unresolved candidate terms.
    Results are cached in Redis/PostgreSQL for 30 days.
    """

    PROMPT = """You are a technical skill taxonomy assistant.
Given a list of candidate terms extracted from a UK tech job description, identify which
ones are genuine technology skills, frameworks, tools, or programming languages.

Return ONLY a JSON array of the terms that ARE genuine tech skills.
If a term is ambiguous or not a skill, exclude it.
Example input:  ["Python", "synergistic", "TensorFlow", "motivated", "LangGraph"]
Example output: ["Python", "TensorFlow", "LangGraph"]

Candidate terms: {terms}
"""

    def __init__(self) -> None:
        from marketforge.memory.redis_cache import LLMCache
        self._cache = LLMCache()

    def resolve(self, candidate_terms: list[str]) -> list[str]:
        if not candidate_terms:
            return []

        cache_key = hashlib.md5("|".join(sorted(candidate_terms)).encode()).hexdigest()
        cached = self._cache.get(f"gate3:{cache_key}")
        if cached:
            return cached.get("skills", [])

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            from marketforge.config.settings import settings

            llm = ChatGoogleGenerativeAI(
                model=settings.llm.fast_model,
                google_api_key=settings.llm.gemini_api_key,
                temperature=0,
            )
            prompt = self.PROMPT.format(terms=json.dumps(candidate_terms[:30]))
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()

            # Clean markdown fences if present
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip()

            skills = json.loads(raw)
            if isinstance(skills, list):
                self._cache.set(f"gate3:{cache_key}", {"skills": skills})
                logger.debug("llm_gate.resolved", count=len(skills), from_candidates=len(candidate_terms))
                return skills
        except Exception as exc:
            logger.warning("llm_gate.error", error=str(exc))
        return []


# ── Three-gate orchestrator ────────────────────────────────────────────────────
_taxonomy  = SkillTaxonomy()
_spacy     = SpacyGate()
_llm_gate  = LLMGate()


def extract_skills(text: str, run_llm_gate: bool = True) -> dict[str, list[tuple[str, str, str, float]]]:
    """
    Run the three-gate extraction pipeline on a job description.

    Returns:
        {
            "gate1": [(canonical, category, "gate1", 1.0), ...],
            "gate2": [(canonical, category, "gate2", 0.9), ...],
            "gate3": [(canonical, "general", "gate3", 0.75), ...],
        }
    """
    results: dict[str, list[tuple[str, str, str, float]]] = {
        "gate1": [], "gate2": [], "gate3": [],
    }

    if not text:
        return results

    # Gate 1: exact taxonomy
    g1 = _taxonomy.extract(text)
    found_canonicals = {skill for skill, _ in g1}
    results["gate1"] = [(skill, cat, "gate1", 1.0) for skill, cat in g1]

    # Gate 2: spaCy
    g2_tokens = _spacy.extract(text, found_canonicals)
    for token in g2_tokens:
        canonical = _taxonomy.resolve(token)
        if canonical and canonical not in found_canonicals:
            found_canonicals.add(canonical)
            cat = _taxonomy._category_map.get(canonical, "general")
            results["gate2"].append((canonical, cat, "gate2", 0.9))

    # Gate 3: LLM on remaining unknown tokens
    if run_llm_gate and g2_tokens:
        unresolved = [t for t in g2_tokens if not _taxonomy.resolve(t)]
        if unresolved:
            g3 = _llm_gate.resolve(unresolved)
            for term in g3:
                if term not in found_canonicals:
                    results["gate3"].append((term, "general", "gate3", 0.75))

    return results


def extract_skills_flat(text: str) -> list[tuple[str, str, str, float]]:
    """Convenience: run all gates and return a flat list."""
    r = extract_skills(text)
    return r["gate1"] + r["gate2"] + r["gate3"]


# ── Salary NER ────────────────────────────────────────────────────────────────
_SALARY_PATTERNS = [
    re.compile(r"£\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?\s*[-–to]+\s*£\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?"),
    re.compile(r"£\s*(\d{1,3}(?:,\d{3})*)\s*(k|K)(?!\+)\b"),  # exclude £Xk+ "at least" vague claims
    re.compile(r"up\s+to\s+£\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?", re.IGNORECASE),
    re.compile(r"from\s+£\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?", re.IGNORECASE),
    re.compile(r"salary\s+of\s+£\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?", re.IGNORECASE),
    re.compile(r"£\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K)?\s*(?:per\s+annum|p\.?a\.?|per\s+year)", re.IGNORECASE),
]


def _parse_salary_value(value_str: str, is_k: bool) -> float:
    cleaned = value_str.replace(",", "").replace("£", "").strip()
    value = float(cleaned)
    if is_k:
        value *= 1000
    return value


def extract_salary(text: str) -> tuple[float | None, float | None]:
    """
    Extract salary range from free-text job description.
    Returns (min, max) in GBP; both None if no salary detected.
    """
    if not text:
        return None, None

    # Pattern 1: range £X[k] – £Y[k]
    m = _SALARY_PATTERNS[0].search(text)
    if m:
        low_s, high_s = m.group(1), m.group(2)
        low_k  = bool(re.search(r'k', m.group(0)[m.group(0).index(m.group(1)) + len(m.group(1)):m.group(0).index(m.group(2))], re.I))
        high_k = bool(re.search(r'k', m.group(0)[m.group(0).index(m.group(2)) + len(m.group(2)):], re.I))
        return _parse_salary_value(low_s, low_k), _parse_salary_value(high_s, high_k)

    # Pattern 2: £Xk shorthand
    m = _SALARY_PATTERNS[1].search(text)
    if m:
        v = _parse_salary_value(m.group(1), True)
        return v, v

    # Patterns 3-6: single value
    for pat in _SALARY_PATTERNS[2:]:
        m = pat.search(text)
        if m:
            raw = m.group(1)
            is_k = bool(re.search(r'k', m.group(0)[m.group(0).index(raw) + len(raw):], re.I))
            v = _parse_salary_value(raw, is_k)
            if v < 1000:   # likely in thousands (£45 → £45k)
                v *= 1000
            # Sanity: UK salaries £15k–£500k
            if 15_000 <= v <= 500_000:
                return v, v

    return None, None


# ── Role Classifier ────────────────────────────────────────────────────────────
_ROLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bml\s+eng|machine\s+learning\s+eng", re.I),                "ml_engineer"),
    (re.compile(r"\bai\s+eng|artificial\s+intelligence\s+eng", re.I),         "ai_engineer"),
    (re.compile(r"\bdata\s+scien", re.I),                                      "data_scientist"),
    (re.compile(r"\bmlops|ml\s+ops|ml\s+platform|ml\s+infra", re.I),         "mlops_engineer"),
    (re.compile(r"\bnlp\s+eng|natural\s+language.*eng", re.I),                "nlp_engineer"),
    (re.compile(r"\bcomputer\s+vision|cv\s+eng|perception\s+eng", re.I),      "computer_vision_engineer"),
    (re.compile(r"\bresearch\s+sci|principal\s+research|staff\s+research", re.I), "research_scientist"),
    (re.compile(r"\bapplied\s+sci", re.I),                                    "applied_scientist"),
    (re.compile(r"\bdata\s+eng", re.I),                                       "data_engineer"),
    (re.compile(r"\bai\s+safety|alignment\s+res", re.I),                      "ai_safety_researcher"),
    (re.compile(r"\bai\s+product|ml\s+product|llm\s+product", re.I),         "ai_product_manager"),
    (re.compile(r"\bllm\s+eng|large\s+language|foundation\s+model", re.I),   "ai_engineer"),
    (re.compile(r"\bgen(erative)?\s+ai\s+eng", re.I),                        "ai_engineer"),
]

_SENIORITY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bjunior|\bjr\b|entry.level|associate\s+(?:ml|ai|data)", re.I), "junior"),
    (re.compile(r"\blead\b|tech\s+lead|engineering\s+lead", re.I),            "lead"),
    (re.compile(r"\bsenior|\bsr\b|staff\b", re.I),                            "senior"),
    (re.compile(r"\bprincipal|\bdirector|\bhead\s+of|\bvp\b", re.I),          "principal"),
]


def classify_role(title: str) -> tuple[str, str]:
    """
    Returns (role_category, experience_level).
    Uses title pattern matching; falls back to 'other' / 'unknown'.
    """
    role = "other"
    for pattern, category in _ROLE_PATTERNS:
        if pattern.search(title):
            role = category
            break

    level = "unknown"
    for pattern, seniority in _SENIORITY_PATTERNS:
        if pattern.search(title):
            level = seniority
            break

    return role, level


# ── Sponsorship & startup detectors ──────────────────────────────────────────
_SPONSOR_POS = [
    re.compile(r"visa\s+sponsor", re.I),
    re.compile(r"willing\s+to\s+sponsor", re.I),
    re.compile(r"skilled\s+worker\s+visa", re.I),
    re.compile(r"tier\s*[23]\s*sponsor", re.I),
    re.compile(r"can\s+(?:provide|offer|support)\s+sponsorship", re.I),
]
_SPONSOR_NEG = [
    re.compile(r"no\s+(?:visa\s+)?sponsor", re.I),
    re.compile(r"cannot\s+(?:offer|provide)\s+sponsor", re.I),
    re.compile(r"uk\s+(?:citizens?\s+only|right\s+to\s+work)", re.I),
    re.compile(r"must\s+have\s+(?:the\s+)?right\s+to\s+work.*without\s+sponsor", re.I),
    re.compile(r"sc\s+clearance|dv\s+clearance|security\s+clearance", re.I),
]
_STARTUP_INDICATORS = [
    re.compile(r"seed\s+(?:stage|funded|round)", re.I),
    re.compile(r"series\s+[abcde]\b", re.I),
    re.compile(r"early.stage|pre.seed|pre-series", re.I),
    re.compile(r"y\s+combinator|yc.backed|a16z|sequoia", re.I),
    re.compile(r"founding\s+(?:engineer|team|member)", re.I),
    re.compile(r"\bstartup\b", re.I),
    re.compile(r"building\s+from\s+(the\s+)?ground", re.I),
]


def detect_sponsorship(text: str) -> tuple[bool | None, bool | None]:
    """Returns (offers_sponsorship, citizens_only)."""
    t = text or ""
    offers  = any(p.search(t) for p in _SPONSOR_POS) or None
    citizens = any(p.search(t) for p in _SPONSOR_NEG) or None
    return offers, citizens


def detect_startup(text: str, company: str = "") -> bool:
    combined = f"{company} {text}"
    return any(p.search(combined) for p in _STARTUP_INDICATORS)


# ── Small lookup set for spaCy ruler ──────────────────────────────────────────
_TECH_PATTERNS = {
    e.lower() for entry in SKILL_TAXONOMY
    for e in ([entry["canonical"]] + entry.get("aliases", []))
}
