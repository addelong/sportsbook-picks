import sys
import unittest
from pathlib import Path
from textwrap import dedent

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

    def test_record_with_spaced_dash(self) -> None:
        record = parse_record("Record: 4 - 0")
        self.assertIsNotNone(record)
        assert record
        self.assertEqual(record.wins, 4)
        self.assertEqual(record.losses, 0)
        self.assertEqual(record.pushes, 0)
        self.assertEqual(record.display, "4 - 0")

    def test_record_with_letter_suffixes(self) -> None:
        record = parse_record("POTD Record: 130w 88l 2p ")
        self.assertIsNotNone(record)
        assert record
        self.assertEqual(record.wins, 130)
        self.assertEqual(record.losses, 88)
        self.assertEqual(record.pushes, 2)
        self.assertEqual(record.display, "130w 88l 2p")


class ParsingRegressionTest(unittest.TestCase):
    def test_billycapezzi_reaves_pick_extracted(self) -> None:
        body = dedent(
            """\
            POTD RECORD: 163-123

            Season record: 0-0

            Last POTD:

            Todays POTD: __Austin Reaves O10.5 RA @2.0__ 1U (Bet365)

            NBA | Lakers vs GSW | 🏀  

            We’re so back fellas! Missed y’all let’s get into it - NBA opening night 🔥

            First of all I’d like to say that this long wait has given me no chance of grabbing a good line without it getting bumped. Even Reaves was at 9.5 last night but I still like the 10.5 so I’m still going for it.

            AR crushed this line in games without LeBron last season and it’s been confirmed for a while that Bron will miss the first few weeks of this season including the opener. Big upside for Reaves as seen by the numbers, he’s gone over this line in 8/9 games without LeBron, 9 straight on 9.5 last season avg 14.9 RA per game on 13.2 potential assists per game and 11.7 rebound chances per game.

            The addition of big man Al Horford for Golden State Warriors will bring more threat from deep for them as his game includes a lot of threes and playing outside the perimeter. New Lakers center Ayton will be dragged out the paint to defend, which will leave lots of rebound chances for the others and as a Lakers fan I know that Reaves is always active to grab them. Self explanatory that the assist upside is there without Bron, as him and Luka will run the plays and both to have a fair share of the ball.

            Like all his lines tomorrow but this one looks the best to me, expecting big big minutes from him in what should be a good game. 

            _RA stands for Rebounds + Assists combined, individual split if you don’t have it would be 5.5 assists imo_
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Lakers vs GSW")
        self.assertEqual(fields["pick"], "Austin Reaves O10.5 RA @2.0")
        self.assertEqual(fields["recommended_wager"], "1U")

    def test_stakesniper_roi_not_selected(self) -> None:
        body = dedent(
            """\
            Record 26 - 24

            ROI +0.52%

            ✅✅❌❌✅✅❌✅❌❌✅❌❌❌❌❌✅✅✅❌✅❌✅✅❌❌✅✅❌✅❌❌✅✅❌✅❌✅❌✅✅✅✅❌❌❌✅❌✅✅

            LAST PICK: Tottenham vs Aston Villa

            ⚽ Pick: Both teams to score 

            💸 Odds: 1.75

            📊 My 50th tip came in nicely ❤️ Bet was won on the 37th minute minute when Aston Villa equalised who then went on to win the game. Would have gotten better odds if I added over 2.5 goals to it but didn't trust that as was expecting a 1-1 draw. Great win nonetheless!

            TODAY'S PICK: Champions League Football ⚽ - Villarreal vs Manchester City (21:00 CET)

            ⚽ Pick: Both teams to score 

            💸 Odds: 1.66

            📊 City’s away games in Europe have consistently produced goals on both sides recently, while Villarreal have a potent attack but also defensive frailties making BTTS highly probable. Villarreal have been strong at home, not losing yet and scoring in all games and have only lost 1 game in all 2025 against Real Madrid.
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Villarreal vs Manchester City")
        self.assertEqual(fields["pick"], "Both teams to score")

    def test_geluksspook24_match_line_retained(self) -> None:
        body = dedent(
            """\
            Record: 5-8 ✅✅❌✅️✅️❌❌❌❌✅️❌❌❌

            Profit: -2.96 units

            Last pick: (Alaves - Valencia) Abde Rebbach over 0.5 shots on target (3.00) ❌

            ------------------------------------------------------------

            Competition: UEFA Champions League

            Match: PSV - Napoli

            Start time: 3pm EST / 21.00 CET

            Pick: **Frank Anguissa over 0.5 shots on target (bookmaker Unibet, odds 2.85)**

            Units: 1
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "PSV - Napoli")
        self.assertEqual(
            fields["pick"],
            "Frank Anguissa over 0.5 shots on target (bookmaker Unibet, odds 2.85)",
        )
        self.assertEqual(fields["recommended_wager"], "1")

    def test_fuzzy_recognition_time_line_ignored(self) -> None:
        body = dedent(
            """\
            Record 5-1 LWWWWW (Will update record)

            Last Pick : NBA Rockets vs Thunder (07:40 ET)
            Reed Sheppard under 8.5 points (1.85) (bet365)

            Todays Pick : NBA Grizzlies vs Pelicans (08:00 ET)
            Grizzlies -4.5 (1.85) (bet365)
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "NBA Grizzlies vs Pelicans (08:00 ET)")
        self.assertEqual(fields["pick"], "Grizzlies -4.5 (1.85) (bet365)")

    def test_geluksspook24_bonus_tip_does_not_replace_game_or_sport(self) -> None:
        body = dedent(
            """\
            Competition: UEFA Champions League

            Match: Galatasaray - Bodo/Glimt

            Pick: Jens Hauge over 0.5 shots on target (bookmaker Unibet, odds 2.25)

            Units: 1

            **BONUS TIP:** He scored 2 headers only five days ago vs Sarpsborg 08
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Galatasaray - Bodo/Glimt")
        self.assertEqual(fields["sport"], "Soccer")

    def test_tokcliff_badminton_pick(self) -> None:
        body = dedent(
            """\
            Event: Korea Masters Women's Single
            POTD Record: 130w 88l 2p
            Date: 3 Nov SGT
            Net Profit = +42.89 units

            Chiu Pin Chian -5.5 points at 1.83 @ 1.5 units (vs Goh Jin Wei)

            Odds on stake. Difference in class, Goh Jin Wei is just bad, and Chiu on okay form. Follow me or not, up to you, i do this for free
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["pick"], "Chiu Pin Chian -5.5 points at 1.83")
        self.assertEqual(fields["game"], "Chiu Pin Chian vs Goh Jin Wei")
        self.assertEqual(fields["recommended_wager"], "1.5 units")
        self.assertEqual(fields["time"], "3 Nov SGT")

    def test_player_prop_odds_does_not_form_matchup(self) -> None:
        body = dedent(
            """\
            Record: 10-5
            Pick: Jamal Murray (Nuggets) O30.5 PRA @ 1.89. 3U play.
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["pick"], "Jamal Murray (Nuggets) O30.5 PRA @ 1.89. 3U play.")
        self.assertIsNone(fields["game"])

    def test_parenthetical_matchup_promoted(self) -> None:
        body = dedent(
            """\
            Today's Pick: Luka Doncic o0.5 Blocks (Spurs vs Lakers)
            Units: 1.0
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["pick"], "Luka Doncic o0.5 Blocks")
        self.assertEqual(fields["game"], "Spurs vs Lakers")
        self.assertEqual(fields["recommended_wager"], "1.0")

    def test_milwaukee_wofford_pick(self) -> None:
        body = dedent(
            """\
            Record: 2-1 (66.7%)

            Today’s Pick: [NCAA Basketball] Milwaukee -1.5 vs Wofford (Caesars -110) 2:00 PM EST

            Once again this is gambling at the end of the day so bet responsibly, control your emotions, and best of luck
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Milwaukee vs Wofford")
        self.assertEqual(fields["pick"], "Milwaukee -1.5 (Caesars -110) 2:00 PM EST")
        self.assertEqual(fields["time"], "2:00 PM EST")
        self.assertEqual(fields["sport"], "NCAA")

    def test_olivecritical_pick_preferred(self) -> None:
        body = dedent(
            """\
            Record:

            ❌❌✅✅✅✅✅

            **Event:** ⚽ / UEFA Champions League / Copenhagen v Dortmund

            **Net Units/ROI:** 6.88u/38.22%

            **Previous Pick:** *Double chance: Mjällby or draw & Over 0,5 goals @ 1,57 -- 2 units.*✅

            **Summary:** The Swedish Leicester City made it happen! After a quick 0-2 lead after 28 minutes they didn't give anything away to become champions. Which is also good as it meant this bet hit. GG's

            **Today's Pick:** ***Dortmund over 1.5 goals @ 1.83 -- 4 units.***
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Copenhagen v Dortmund")
        self.assertEqual(fields["pick"], "Dortmund over 1.5 goals @ 1.83")
        self.assertEqual(fields["recommended_wager"], "4 units")

    def test_vander_chill_comment_is_skipped(self) -> None:
        body = "At 43-22 you are still better than most... a few losses is just mean reversion.  I like both picks... welcome back!"
        fields = extract_pick_fields(body.splitlines())
        self.assertIsNone(fields["pick"])
        self.assertIsNone(fields["game"])

    def test_sport_not_extracted_from_record_breakdown(self) -> None:
        body = dedent(
            """\
            **POTD Record: 11-4 (+23.43 Units)**

            **E-Sports 9-1 | NBA 2-1 | NFL 0-2**

            **Event:** Pro Dart World Championship

            **Pick:** Wesley Plaisier +1.5 vs. Krzysztof Ratajski @ 1.67 (5 Units)
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Darts")
        self.assertIsNotNone(fields["pick"])

    def test_boxing_day_does_not_extract_boxing_sport(self) -> None:
        body = dedent(
            """\
            **POTD Record:** 101 – 66 - 10

            **New Event:** EPL – Nottingham Forest vs Manchester City

            **Pick:**  Manchester City -1 @ 1.88 (Usual 3 units)

            The day after Boxing Day just happens to fall on a Saturday.
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertIn(fields["sport"], ["Soccer", "EPL"])
        self.assertIn("Nottingham Forest", fields["game"])
        self.assertIn("Manchester City", fields["game"])

    def test_time_string_not_extracted_as_sport(self) -> None:
        body = dedent(
            """\
            Record: 2-4

            Event: England Premier League | 12:30 PM EST

            POTD: Aston Villa Double Chance vs Chelsea @ 1.99 odds (3u)
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Soccer")
        self.assertEqual(fields["pick"], "Aston Villa Double Chance")
        self.assertIn("Chelsea", fields["game"])

    def test_leading_and_stripped_from_game(self) -> None:
        body = dedent(
            """\
            Record: 2-2

            Soccer | Premier League
            Matches: Arsenal vs Brighton & Hove Albion
            and Nottingham Forest vs Manchester City
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Soccer")
        game = fields.get("game") or ""
        self.assertFalse(game.lower().startswith("and "))

    def test_hockey_extracted_from_parenthetical(self) -> None:
        body = dedent(
            """\
            **POTD Record** : 8-4

            **Today's pick** : Ottawa Senators vs Toronto Maple Leafs at 7:00pm EST (NHL/Hockey)

            Brady Tkachuk over 0.5 points (-175)
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Hockey")
        self.assertEqual(fields["time"], "7:00pm EST")

    def test_frozenstride_followup_pick_used(self) -> None:
        body = dedent(
            """\
            **Record:** 2-2 *(+0.3)*

            **Todays Bet:** Union Saint Gilloise - Inter 
            **Inter WIN**

            **Odds:** $1.75

            **Wager:** 2 units 

            **When:** 11 hours from post

            **Why:** 

            Inter Milan should stroll into Brussels with all the quiet confidence of a team that’s been here and done it a hundred times before, because, they basically have. Union Saint Gilloise are a tidy outfit with plenty of heart sure, but they’re about to find out that Serie A’s best don’t do charity work. Even with Marcus Thuram sidelined, Inter have too much quality all over the pitch. 

            Lautaro Martinez is scoring like a well oiled scoring machine, and Barella’s engine never wants to stop. Union might try to press high and ride the home crowd energy, but once Inter settle in and start warming up and passing it around, it should put a stop to the noise of the fans. Expect a professional, nononsense display from Inter, think less chaos, more cold efficiency. A comfortable 2-0 or 3-1 Inter win feels about right.
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Union Saint Gilloise - Inter")
        self.assertEqual(fields["pick"], "Inter WIN")
        self.assertEqual(fields["recommended_wager"], "2 units")

    def test_the_kendawg_pick_not_overwritten(self) -> None:
        body = dedent(
            """\
            🎯 **Pick: Manchester City –1.5 AH**

            Event: UEFA Champions League 

            Odds: 2.75 (Bet365) 

            Model Fair: 2.3

            Stake: $100 (2 Units) 

            Record: 2-2

            Poisson/Elo Model interpretation: 

            expected goals:

            λ City ≈ 1.90  λ Villarreal ≈ 0.8

            From simulated score distributions, City win by 2+ ≈ 43 % of the time

            Fair odds = 1 / 0.43 ≈ 2.33

            Book price = 2.75 +EV (~+18 %).
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["pick"], "Manchester City -1.5 AH @ 2.75 (Bet365)")
        self.assertEqual(fields["recommended_wager"], "2 Units")

    def test_reptilia_pick_trailing_bar_removed(self) -> None:
        body = "POTD: Kecmanovic vs Wawrinka | Pick: Kecmanovic ML (-175) | Units: 4u"
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Kecmanovic vs Wawrinka")
        self.assertEqual(fields["pick"], "Kecmanovic ML (-175)")
        self.assertEqual(fields["recommended_wager"], "4u")

    def test_successful_wall_split_game_and_bet(self) -> None:
        body = dedent(
            """\
            POTD: Ipswich Town vs Charlton Athletic ( Ipswich ML + O1.5 Goals ) @-102 2u
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["game"], "Ipswich Town vs Charlton Athletic")
        self.assertEqual(fields["pick"], "ML + O1.5 Goals @-102")
        self.assertEqual(fields["recommended_wager"], "2u")

    def test_epl_with_previous_and_new_event(self) -> None:
        body = dedent(
            """\
            **POTD Record:** 101 – 66 - 10

            **Previous:**  EPL – Manchester United vs Newcastle Draw no Bet – LOST

            **Recap:** Early goal by United defended until the end.

            **New Event:** EPL – Nottingham Forest vs Manchester City

            **Pick:**  Manchester City -1 @ 1.88 (Usual 3 units)
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "EPL")
        self.assertEqual(fields["game"], "Nottingham Forest vs Manchester City")
        self.assertEqual(fields["pick"], "Manchester City -1 @ 1.88")

    def test_hockey_league_line_not_treated_as_matchup(self) -> None:
        body = dedent(
            """\
            Record: 34-15

            International Hockey U20 - World Junior Championship

            Pick: Czechia vs. Denmark | Czechia -4.5 (-130)

            Time: 8:30pm EST
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Hockey")
        self.assertEqual(fields["game"], "Czechia vs Denmark")
        self.assertEqual(fields["pick"], "Czechia -4.5 (-130)")

    def test_nfl_with_football_emoji(self) -> None:
        body = dedent(
            """\
            **Record**: 20-12

            **Net Units**: +7.01

            **Todays Event** 🏈 NFL Saturday Night Football  Ravens @ Packers  7:15 PM CST 🏈

            **Todays Pick**  Emanuel Wilson o30.5 rushing yards (-112 on DK) 1u to win 0.89u
            """
        )
        fields = extract_pick_fields(body.splitlines())
        # Should extract sport from emoji or NFL mention
        self.assertIn(fields["sport"], ["Football", "NFL"])
        self.assertIn("Ravens", fields["game"])
        self.assertIn("Packers", fields["game"])

    def test_nba_team_name_detection(self) -> None:
        body = dedent(
            """\
            **POTD Record:** 101-66-10
            **Todays Pick:** New York Knicks vs Atlanta Hawks
            **Pick:** Knicks -3.5 @ 1.90
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "NBA")
        self.assertIn("Knicks", fields["game"])
        self.assertIn("Hawks", fields["game"])

    def test_nfl_team_name_detection(self) -> None:
        body = dedent(
            """\
            **Record:** 15-8
            **Pick:** Texans @ Chargers
            **Bet:** Chargers -4 (-110)
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "NFL")
        self.assertIn("Texans", fields["game"])
        self.assertIn("Chargers", fields["game"])

    def test_soccer_team_name_detection(self) -> None:
        body = dedent(
            """\
            **Record:** 20-10
            **Game:** Liverpool vs Wolves
            **Pick:** Liverpool to win @ 1.50
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Soccer")
        self.assertIn("Liverpool", fields["game"])
        self.assertIn("Wolves", fields["game"])

    def test_slash_separated_sport_prefix(self) -> None:
        body = dedent(
            """\
            **Record:** 25-15
            **Previous:** NCAAB / UConn vs Villanova - WON
            **Pick:** NCAAB / Stanford vs CSUN
            **Bet:** Stanford -12.5
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "NCAAB")
        self.assertIn("Stanford", fields["game"])
        self.assertIn("CSUN", fields["game"])

    def test_pick_not_replaced_with_units_line(self) -> None:
        """Regression test: Don't replace complete pick with 'to win' from units line."""
        body = dedent(
            """\
            NBA Basketball - Saturday 12/27/25

            Pick of the Day: Denver Nuggets -4.5 (-115) @ Orlando Magic

            1 unit to win 0.87 units

            Record: 3-2
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "NBA")
        self.assertIn("Denver Nuggets", fields["game"])
        self.assertIn("Orlando Magic", fields["game"])
        # The key assertion: pick should contain the spread, not just "to win"
        self.assertIn("-4.5", fields["pick"])
        self.assertIn("Nuggets", fields["pick"])
        self.assertNotEqual(fields["pick"], "to win")

    def test_todays_pick_not_last_pick(self) -> None:
        """Regression test: Extract Today's pick, not Last pick."""
        body = dedent(
            """\
            Overall record 75W-49L

            Last pick:
            Manchester United Vs Newcastle

            Matheus Cunha 2 or more shots on target.

            (2.10) 2.5 units ✖️


            Today's pick:

            Premier League

            Liverpool Vs Wolves

            Liverpool to win and Virgil Van Djik 1 or more shot

            1.80 3 units
            """
        )
        fields = extract_pick_fields(body.splitlines())
        # The key assertion: game should be Liverpool vs Wolves, NOT Manchester United vs Newcastle
        self.assertIn("Liverpool", fields["game"])
        self.assertIn("Wolves", fields["game"])
        self.assertNotIn("Manchester United", fields["game"])
        self.assertNotIn("Newcastle", fields["game"])

    def test_a_league_detected_as_soccer(self) -> None:
        """Regression test: A-League Men should be detected as Soccer."""
        body = dedent(
            """\
            **Event:** A-League Men, Central Coast Mariners v Brisbane Roar

            **POTD:** Central Coast Mariners Win or Draw -125
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Soccer")
        self.assertIn("Central Coast Mariners", fields["game"])
        self.assertIn("Brisbane Roar", fields["game"])

    def test_sport_emoji_exits_previous_pick_block(self) -> None:
        """Regression test: Sport emoji line should exit previous pick block."""
        body = dedent(
            """\
            Last POTD: Celtics -8.5 (-110)

            🏀NCAAM🏀

            Today's Event: Witchita State @ UAB

            POTD: UAB ML (-160)
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Ncaam")
        self.assertIn("UAB", fields["game"])
        self.assertIn("-160", fields["pick"])

    def test_sport_prefix_line_extracts_game(self) -> None:
        """Regression test: 'Sport : NCAA Football Team vs Team' extracts both sport and game."""
        body = dedent(
            """\
            Last pick: Previous game

            Sport : NCAA Football  Miami FL vs Ohio State 7:30pm EST

            Pick: Ohio State team total over 101.5
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "Football")
        self.assertIn("Miami", fields["game"])
        self.assertIn("Ohio State", fields["game"])

    def test_pick_line_with_sport_token_sets_sport(self) -> None:
        """Regression test: 'Pick: NCAAB' followed by game sets sport correctly."""
        body = dedent(
            """\
            Pick: NCAAB
            South Alabama vs UL Lafayette

            UNDER 127.5
            """
        )
        fields = extract_pick_fields(body.splitlines())
        self.assertEqual(fields["sport"], "NCAAB")
        self.assertIn("South Alabama", fields["game"])
        self.assertIn("UNDER", fields["pick"])


if __name__ == "__main__":
    unittest.main()
