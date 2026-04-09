"""
MarketForge AI — Locust Load Test.

Target: P95 response time < 8 seconds under realistic traffic.

Usage:
    pip install locust
    locust -f tests/load/locustfile.py --host http://localhost:8000

    # Headless CI run (30s, 10 users, ramp 2/s):
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
           --headless -u 10 -r 2 --run-time 30s \
           --csv tests/load/results

Scenarios:
  MarketUser   — read-heavy (health, skills, salary, trending, snapshot)
  CareerUser   — career/analyse POST (LLM-backed, expensive)
  CVUser       — cv-analyse POST with synthetic PDF (rate-limited 3/hr)

Task weights reflect expected production traffic split:
  80% market reads · 15% career analyse · 5% CV upload
"""
from __future__ import annotations

import io
import zipfile

from locust import HttpUser, between, task


# ── Synthetic CV fixtures ─────────────────────────────────────────────────────

def _make_pdf() -> bytes:
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
        b"Senior ML Engineer. Python PyTorch Docker MLflow.\n"
        b"Experience\nLead ML Engineer 2020-2024\n"
        b"Skills\nPython PyTorch scikit-learn SQL Docker\n"
        b"Education\nMSc Computer Science 2019\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n9\n%%EOF\n"
    )


def _make_docx() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml"'
            ' ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        zf.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1"'
            ' Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
            ' Target="word/document.xml"/>'
            "</Relationships>",
        )
        paras = "".join(
            f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>'
            for p in [
                "Senior ML Engineer",
                "Python PyTorch scikit-learn SQL Docker MLflow FastAPI",
                "Lead ML Engineer at DeepMind 2020-2024",
                "MSc Computer Science UCL 2019",
            ]
        )
        zf.writestr(
            "word/document.xml",
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{paras}</w:body></w:document>".encode(),
        )
        zf.writestr(
            "word/_rels/document.xml.rels",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>'
        )
    return buf.getvalue()


_PDF_BYTES  = _make_pdf()
_DOCX_BYTES = _make_docx()

_CAREER_PAYLOAD = {
    "skills": ["Python", "PyTorch", "MLflow", "Docker", "SQL"],
    "target_role": "Machine Learning Engineer",
    "experience_level": "mid",
    "location": "London",
    "visa_sponsorship": False,
}


# ── User classes ──────────────────────────────────────────────────────────────

class MarketUser(HttpUser):
    """
    Read-heavy user: hits market data endpoints.
    Weight 8 → 80% of virtual users.
    Target: P95 < 500ms (Redis-cached).
    """
    weight      = 8
    wait_time   = between(1, 3)

    @task(4)
    def health(self):
        self.client.get("/api/v1/health")

    @task(3)
    def skills(self):
        self.client.get("/api/v1/market/skills?role_category=all")

    @task(2)
    def trending(self):
        self.client.get("/api/v1/market/trending?days=7")

    @task(2)
    def salary(self):
        self.client.get("/api/v1/market/salary?role_category=all&experience_level=mid&location=London")

    @task(1)
    def snapshot(self):
        self.client.get("/api/v1/market/snapshot")

    @task(1)
    def jobs(self):
        self.client.get("/api/v1/jobs?page=1&page_size=10")


class CareerUser(HttpUser):
    """
    Career analysis user: LLM-backed endpoint.
    Weight 1.5 → 15% of virtual users.
    Target: P95 < 8s.
    """
    weight    = 2
    wait_time = between(5, 15)   # polite — LLM calls are expensive

    @task
    def career_analyse(self):
        with self.client.post(
            "/api/v1/career/analyse",
            json=_CAREER_PAYLOAD,
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                resp.success()   # rate limit is expected behaviour, not a failure
            elif resp.status_code not in (200, 429):
                resp.failure(f"Unexpected status {resp.status_code}")


class CVUser(HttpUser):
    """
    CV upload user: most expensive endpoint (scan + parse + ATS + LLM).
    Weight 0.5 → 5% of virtual users.
    Target: P95 < 8s.
    """
    weight    = 1
    wait_time = between(20, 60)  # very polite — 3/hour real rate limit

    @task(1)
    def cv_pdf(self):
        with self.client.post(
            "/api/v1/career/cv-analyse",
            params={"target_role": "ml_engineer", "consent": "true"},
            files={"cv_file": ("cv.pdf", _PDF_BYTES, "application/pdf")},
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                resp.success()
            elif resp.status_code not in (200, 429):
                resp.failure(f"CV PDF: unexpected status {resp.status_code}")

    @task(1)
    def cv_docx(self):
        with self.client.post(
            "/api/v1/career/cv-analyse",
            params={"target_role": "data_scientist", "consent": "true"},
            files={"cv_file": ("cv.docx", _DOCX_BYTES,
                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                resp.success()
            elif resp.status_code not in (200, 429):
                resp.failure(f"CV DOCX: unexpected status {resp.status_code}")

    @task(1)
    def cv_no_consent_rejected(self):
        """Verify 403 is returned without consent — counts as a success."""
        with self.client.post(
            "/api/v1/career/cv-analyse",
            params={"target_role": "ml_engineer", "consent": "false"},
            files={"cv_file": ("cv.pdf", _PDF_BYTES, "application/pdf")},
            catch_response=True,
            name="/api/v1/career/cv-analyse [no-consent]",
        ) as resp:
            if resp.status_code == 403:
                resp.success()
            else:
                resp.failure(f"Expected 403, got {resp.status_code}")
