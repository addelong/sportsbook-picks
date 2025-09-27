import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pick_collector import extract_pick_fields, parse_pick_text, parse_record  # type: ignore


class ExtractPickFieldsTest(unittest.TestCase):
    def test_pick_line_populates_game_and_pick(self) -> None:
        body = """Record: 12-5\nPick: 2u - SF Giants vs LA Dodgers - Under 8 (-110)\nSport: MLB\nUnits: 2u\n"""
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "SF Giants vs LA Dodgers")
        self.assertEqual(fields["pick"], "Under 8 (-110)")
        self.assertEqual(fields["recommended_wager"], "2u")

    def test_trailing_parenthetical_units_are_captured(self) -> None:
        body = """Record: 7-3\nPick: Brewers ML (-130) (1.5u)\nSport: MLB\n"""
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["pick"], "Brewers ML (-130)")
        self.assertEqual(fields["recommended_wager"], "1.5u")

    def test_potd_line_is_captured(self) -> None:
        body = """POTD Record 25-8 (4 pushes)\n\nTodays POTD: Crystal Palace vs Liverpool - Liverpool to win. Odds 1.90 UK time: 15:00"""
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Crystal Palace vs Liverpool")
        self.assertEqual(fields["pick"], "Liverpool to win. Odds 1.90 UK time: 15:00")

    def test_parlay_pick_retains_full_detail(self) -> None:
        body = """POTD: Real Madrid or draw vs Atletico + PSG ML vs Auxerre @1.65 5U"""
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["pick"], "Real Madrid or draw vs Atletico + PSG ML vs Auxerre @1.65")
        self.assertEqual(fields["recommended_wager"], "5U")

    def test_wager_amount_line_is_cleaned(self) -> None:
        body = """Season Record: 10-3\nToday's Pick: Team A vs Team B - Team A ML\nWager Amount: 1.5 units"""
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["recommended_wager"], "1.5 units")


class ParsePickTextTest(unittest.TestCase):
    def test_prefix_units_are_cleaned(self) -> None:
        game, detail, stake = parse_pick_text("1U - Rangers -1.5 (-120)")
        self.assertEqual(game, "Rangers")
        self.assertEqual(detail, "-1.5 (-120)")
        self.assertEqual(stake, "1U")


class ParseRecordTest(unittest.TestCase):
    def test_record_with_parenthetical_pushes(self) -> None:
        record = parse_record("POTD Record 25-8 (4 pushes)")
        self.assertIsNotNone(record)
        assert record
        self.assertEqual(record.wins, 25)
        self.assertEqual(record.losses, 8)
        self.assertEqual(record.pushes, 4)
        self.assertEqual(record.display, "25-8 (4 pushes)")


if __name__ == "__main__":
    unittest.main()
