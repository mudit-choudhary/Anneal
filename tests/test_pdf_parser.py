"""Unit tests for word-to-region assignment (no GPU or model required)."""

from pdf_parser import assign_words_to_regions, words_to_lines


def region(label, x0, y0, x1, y1, conf=0.9):
    return {"label": label, "conf": conf, "bbox": [x0, y0, x1, y1]}


def word(x0, y0, x1, y1, text, block_no=0):
    return (x0, y0, x1, y1, text, block_no, 0, 0)


class TestAssignment:
    def test_word_goes_to_containing_region(self):
        regions = [region("Text", 0, 0, 100, 100)]
        out, swallowed = assign_words_to_regions([word(10, 10, 30, 20, "hello")], regions)
        assert out[0]["words"][0][4] == "hello"
        assert swallowed == []

    def test_smallest_region_wins(self):
        # A Caption box sitting inside a Picture box claims its own words.
        pic = region("Picture", 0, 0, 300, 300)
        cap = region("Caption", 20, 250, 280, 290)
        out, swallowed = assign_words_to_regions(
            [word(30, 260, 60, 280, "Figure"), word(150, 100, 170, 110, "axis")],
            [pic, cap],
        )
        cap_out = next(r for r in out if r["label"] == "Caption")
        assert [w[4] for w in cap_out["words"]] == ["Figure"]
        # The word inside only the Picture is swallowed, not kept in the flow.
        assert [s["text"] for s in swallowed] == ["axis"]

    def test_uncovered_words_become_fallback_text(self):
        regions = [region("Text", 0, 0, 100, 100)]
        out, _ = assign_words_to_regions(
            [word(500, 500, 520, 510, "stray", block_no=3)], regions)
        fallback = [r for r in out if r.get("fallback")]
        assert len(fallback) == 1
        assert fallback[0]["label"] == "Text"
        assert fallback[0]["words"][0][4] == "stray"

    def test_header_text_swallowed(self):
        regions = [region("Page-header", 0, 0, 600, 30)]
        out, swallowed = assign_words_to_regions([word(10, 5, 80, 25, "Running")], regions)
        assert swallowed[0]["label"] == "Page-header"
        assert all(not r.get("words") for r in out)


class TestWordsToLines:
    def test_lines_in_reading_order(self):
        words = [
            [10, 20, 40, 30, "world"],
            [0, 20, 9, 30, "hello"],
            [0, 40, 30, 50, "second"],
        ]
        assert words_to_lines(words) == ["hello world", "second"]

    def test_slight_baseline_jitter_same_line(self):
        words = [
            [0, 20.0, 30, 30.0, "left"],
            [40, 21.5, 70, 31.5, "right"],
        ]
        assert words_to_lines(words) == ["left right"]

    def test_empty(self):
        assert words_to_lines([]) == []
