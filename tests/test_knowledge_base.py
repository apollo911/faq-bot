from types import SimpleNamespace

import pytest

import knowledge_base
from knowledge_base import KnowledgeBaseError, load_knowledge_base


def test_reads_txt_files(tmp_path):
    (tmp_path / "faq.txt").write_text("Доставка за 1 день.", encoding="utf-8")

    result = load_knowledge_base(tmp_path)

    assert '<document name="faq.txt">' in result
    assert "Доставка за 1 день." in result


def test_reads_pdf_files(tmp_path, monkeypatch):
    (tmp_path / "prices.pdf").write_bytes(b"%PDF-stub")
    fake_pages = [SimpleNamespace(extract_text=lambda: "Страница 1"),
                  SimpleNamespace(extract_text=lambda: "Страница 2")]
    monkeypatch.setattr(knowledge_base, "PdfReader", lambda path: SimpleNamespace(pages=fake_pages))

    result = load_knowledge_base(tmp_path)

    assert '<document name="prices.pdf">' in result
    assert "Страница 1\nСтраница 2" in result


def test_ignores_unsupported_files(tmp_path):
    (tmp_path / "faq.txt").write_text("Ответ", encoding="utf-8")
    (tmp_path / "notes.docx").write_text("Не должно попасть", encoding="utf-8")

    result = load_knowledge_base(tmp_path)

    assert "Не должно попасть" not in result


def test_files_are_read_in_name_order(tmp_path):
    (tmp_path / "b.txt").write_text("второй", encoding="utf-8")
    (tmp_path / "a.txt").write_text("первый", encoding="utf-8")

    result = load_knowledge_base(tmp_path)

    assert result.index("первый") < result.index("второй")


def test_does_not_modify_files(tmp_path):
    path = tmp_path / "faq.txt"
    path.write_text("Ответ", encoding="utf-8")
    modified_before = path.stat().st_mtime_ns

    load_knowledge_base(tmp_path)

    assert path.read_text(encoding="utf-8") == "Ответ"
    assert path.stat().st_mtime_ns == modified_before


def test_empty_folder_raises(tmp_path):
    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(tmp_path)


def test_blank_files_raise(tmp_path):
    (tmp_path / "faq.txt").write_text("   \n", encoding="utf-8")

    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(tmp_path)


def test_missing_folder_raises(tmp_path):
    with pytest.raises(KnowledgeBaseError):
        load_knowledge_base(tmp_path / "missing")


def test_real_knowledge_base_loads():
    from bot import KNOWLEDGE_DIR

    result = load_knowledge_base(KNOWLEDGE_DIR)

    assert "ТехноДом" in result
