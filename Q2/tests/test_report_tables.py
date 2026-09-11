import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emergency_table import target_indices
from fill_reports import replace_table, validate_markdown_tables
from make_results import build_emergency_rows


class ReportTableTests(unittest.TestCase):
    def test_repeated_dates_in_other_sections_are_preserved(self):
        text = "### Plan\n\n| Date | Energy |\n|---|---|\n| 2025.3.20 | old |\n\n### Emergency\n\n| Date | Period | Energy |\n|---|---|---|\n| 2025.3.20 | old | old |\n"
        replacement = "| Date | Energy |\n|---|---|\n| 2025.3.20 | 12 |"
        changed = replace_table(text, "### Plan", replacement)
        self.assertEqual(changed.split("### Emergency")[1], text.split("### Emergency")[1])
        validate_markdown_tables(changed)

    def test_emergency_gaps_stay_separate_and_day_end_is_24(self):
        e = np.zeros((1, 144))
        e[0, 0:2] = [2.0, 3.0]
        e[0, 143] = 7.0
        rows = build_emergency_rows(np.array([45736.]), e)
        self.assertEqual(rows[1], [45736., "0:00-0:20", 5.0])
        self.assertEqual(rows[2], [None, "23:50-24:00", 7.0])

    def test_requested_dates_are_exact(self):
        dates = np.arange(45689., 46023.)
        self.assertEqual(target_indices(dates), [47, 140, 234, 323])

    def test_malformed_table_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_markdown_tables("| Date | Amount |\n|---|---|\n| date | wrong | extra |\n")


if __name__ == "__main__":
    unittest.main()
