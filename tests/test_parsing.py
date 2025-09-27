import unittest

from pick_collector import extract_pick_fields, parse_pick_text


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


class ParsePickTextTest(unittest.TestCase):
    def test_prefix_units_are_cleaned(self) -> None:
        game, detail, stake = parse_pick_text("1U - Rangers -1.5 (-120)")
        self.assertEqual(game, "Rangers")
        self.assertEqual(detail, "-1.5 (-120)")
        self.assertEqual(stake, "1U")


if __name__ == "__main__":
    unittest.main()
