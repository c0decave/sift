"""PIN and TAN are words of their own, not three letters inside another.

A review read /usr/lib/python3.14 through find_secrets once quoted keys
were read (0.1.76) and found the country tables of the standard library
masked:

    'Pakistan': 'PK',
    'Philippines': 'PH',
    'alpine': 'reboot',

"tan" and "pin" were two of the secret-name rule's words with no boundary
in front (_NAMES_A_SECRET), and "pin" had none in the sentence rule either
(_SECRET_IN_SENTENCE): Pakistan ends in "tan", Philippines holds "pin" and
the plural "es", alpine holds "pin" and "e". Every value under such a key
was a password to the strong tier, and a host application then hid "PK", "PH" and
"reboot" wherever a model's answer said them.

The two are words of a key where a key's words begin and end: at its
start or end, at a separator (user_pin, PIN_NUMBER, TAN_LIST), and at a
camelCase hump (pinCode, userPin, mTAN, pushTAN) -- the same places
_KEY_WORDS splits a key. Glued into a longer lowercase word they are no
key's word: spinner, Tanzania, standard, important, opinion, shipping.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.extract import find_secrets
from sift.extract.base import (_secret_key, looks_like_a_credential,
                               redact_value)

#: A value no rule hides for its own shape: only a key naming a secret can.
SECRET = "Kx9vQ2mZ7pL4Wq8rTz"

#: Keys that name a PIN or a TAN, as the files sift reads write them.
NAMES = ("PIN", "pin", "Pin", "user_pin", "userPin", "userPIN", "pinCode",
         "PINCode", "pin_code", "PIN_NUMBER", "ALARM_PIN", "db.pin", "pins",
         "PINs", "tan", "TAN", "TAN_LIST", "tan-liste", "mTAN", "MTAN",
         "iTAN", "eTAN", "pushTAN", "smsTAN", "chipTAN", "photoTAN", "tans",
         "TANs", "sms_tan", "pushtan")

#: Keys that hold "pin" or "tan" inside a word of their own. Each of the
#: first eight was a secret's name to at least one rule before this.
WORDS = ("Pakistan", "Philippines", "alpine", "spinner", "Tanzania",
         "standard", "important", "Titan", "spine", "pine", "Afghanistan",
         "Kazakhstan", "sultan", "Tibetan", "Montana", "tangent", "opinion",
         "pinned", "shipping", "mapping", "pinkie", "atan2", "SPINNER",
         "ALPINE_VERSION", "PAKISTAN", "Spain", "Stand", "Tanne")


def found(text: str) -> "list[str]":
    """The texts find_secrets masks in ``text``, bidi controls aside."""
    return [text[span.start:span.end]
            for span in find_secrets(text) if span.reason != "bidi"]


def shapes(key: str, value: str) -> dict:
    """The key beside its value as a dict literal, a JSON member and an
    assignment line write it."""
    return {"quoted": f"    '{key}': '{value}',\n",
            "json": f'  "{key}": "{value}",\n',
            "assignment": f"{key} = {value}\n"}


class ShortNamesAreWordsTest(unittest.TestCase):

    def test_a_pin_or_a_tan_names_a_secret(self):
        """Happy path: under each name the value is a secret to the key
        rule, the settings rule and every shape find_secrets reads."""
        for key in NAMES:
            with self.subTest(key=key):
                self.assertTrue(looks_like_a_credential(SECRET, key))
                self.assertTrue(_secret_key(key))
                for shape, text in shapes(key, SECRET).items():
                    self.assertEqual([SECRET], found(text), shape)

    def test_a_pin_is_hidden_in_a_sentence(self):
        """What scrub reads -- a summary, a note, a model's answer --
        hides the value after the name too; mTAN, pushTAN and TANs were
        no key to the sentence rule, whose "tan" wanted no letter on
        either side."""
        for key in ("PIN", "pinCode", "PIN_NUMBER", "tan", "TAN_LIST",
                    "mTAN", "pushTAN", "TANs", "iTAN"):
            with self.subTest(key=key):
                said = redact_value(f"Die {key} lautet {SECRET}.")
                self.assertNotIn(SECRET, said)
                self.assertIn(key, said)

    def test_a_word_holding_pin_or_tan_names_nothing(self):
        """Sad path: glued into a longer word, "pin" and "tan" name no
        secret; the value stays, whatever it looks like."""
        for key in WORDS:
            with self.subTest(key=key):
                self.assertFalse(looks_like_a_credential(SECRET, key))
                self.assertFalse(_secret_key(key))
                for shape, text in shapes(key, SECRET).items():
                    self.assertEqual([], found(text), shape)
                sentence = f"{key}: {SECRET}"
                self.assertEqual(sentence, redact_value(sentence))

    def test_the_standard_library_tables_found(self):
        """The lines themselves, as /usr/lib/python3.14 writes them."""
        for text in ("    'Pakistan': 'PK',\n", "    'Philippines': 'PH',\n",
                     "    'alpine': 'reboot',\n", "    'Afghanistan': 'AF',\n",
                     '    "Kazakhstan": "KZ",\n'):
            with self.subTest(text=text):
                self.assertEqual([], found(text))
                self.assertEqual(text, redact_value(text))

    def test_a_certificate_line_names_no_tan(self):
        """base64 is letters of both cases, and an RSA certificate's
        algorithm, "...nTANBgkqhkiG9w0BAQ...", reads as a hump before TAN
        and one after it: the line after it was a TAN's value to the
        sentence rule (8 lines of /etc/ca-certificates) until a word may
        end only where a small letter meets a capital."""
        pem = ("-----BEGIN " "CERTIFICATE-----\n"
               "MIIEzDCCA7SgAwIBAgIQPzXk3f9QpLmZ0aB7cnTANBgkqhkiG9w0BAQsFADBx\n"
               "QzELMAkGA1UEBhMCREUxEzARBgNVBAgMClNvbWUtU3RhdGUxITAfBgNVBAoM\n"
               "-----END CERTIFICATE-----\n")
        self.assertEqual([], found(pem))
        self.assertEqual(pem, redact_value(pem))
        # "...iMTAnMAoG...": a hump, then an m and a TAN in two cases -- a
        # word is written in one (_PIN_OR_TAN).
        pem = pem.replace("QzELMAkGA1UEBhMC", "dOGiMTAnMAoGCCsG")
        self.assertEqual([], found(pem))
        self.assertFalse(looks_like_a_credential(SECRET, "iMTAnMAoG"))

    def test_a_setting_under_a_word_keeps_its_plain_value(self):
        """The settings rule withheld every plain value under a key it took
        for a secret's name: ALPINE_VERSION=3.20 was a PIN's."""
        from sift.extract.base import _harmless_setting
        self.assertTrue(_harmless_setting("ALPINE_VERSION", "3.20"))
        self.assertTrue(_harmless_setting("SPINNER_STYLE", "dots"))
        self.assertFalse(_harmless_setting("SIM_PIN", "4711"))

    def test_the_edges_of_a_word(self):
        """Edge cases: a hump after an acronym (APIPin), a digit or a
        separator after the name, the exemptions of a name for a secret,
        and a PIN's plural against a pine's."""
        for key in ("APIPin", "SIMPin", "PIN_2", "pin.code", "TAN_1"):
            with self.subTest(key=key):
                self.assertTrue(looks_like_a_credential(SECRET, key))
        # Named *for* a secret, not holding one: the suffix exemption still
        # decides first.
        for key in ("pin_required", "tan_length", "PIN_MAX_ATTEMPTS"):
            with self.subTest(key=key):
                self.assertFalse(looks_like_a_credential(SECRET, key))
        for key in ("pines", "tane", "tanes", "Pines"):
            with self.subTest(key=key):
                self.assertFalse(looks_like_a_credential(SECRET, key))


#: Keys that name a one-time password, a pre-shared key, a seed, a DSN or
#: MFA as the files sift reads write them: at the start or end of the key,
#: beside a separator, at a hump.
OTHER_NAMES = ("otp", "OTP", "user_otp", "userOtp", "OTP_CODE", "otp.code",
               "totp", "TOTP", "totp_seed", "mfa", "MFA", "mfa_token",
               "psk", "PSK", "wifi_psk", "wpaPsk", "WPA_PSK", "WPAPsk",
               "seed", "SEED", "wallet_seed", "walletSeed", "seedPhrase",
               "seeds", "dsn", "DSN", "db_dsn", "sentryDsn", "SENTRY_DSN",
               "dsn.url")

#: Keys that hold one of those inside a word of their own: each named a
#: secret to the strong tier (_NAMES_A_SECRET) or the sentence rule
#: (_SECRET_IN_SENTENCE) before this.
OTHER_WORDS = ("linseed", "aniseed", "rapeseed", "flaxseed", "birdseed",
               "cottonseed", "linseed_oil", "LINSEED", "cropsk", "odsn",
               "footprint", "spotprice", "notpad", "BOOTPROTO", "rootpath",
               "despotism", "Footprint", "FOOTPRINT")


class OtherShortNamesAreWordsTest(unittest.TestCase):
    """otp, totp, mfa, psk, seed and dsn as PIN and TAN are: words of their
    own. The strong tier read them with no boundary in front, so linseed,
    aniseed and rapeseed were a seed, cropsk a pre-shared key and odsn a
    DSN -- every value under them a password --; the sentence rule read
    "otp" with no boundary at all, so "footprint: 4096" and "spotprice =
    42.10" hid their numbers."""

    def test_each_name_names_a_secret(self):
        """Happy path, as before: the value is a secret under each name,
        in every shape find_secrets reads."""
        for key in OTHER_NAMES:
            with self.subTest(key=key):
                self.assertTrue(looks_like_a_credential(SECRET, key))
                self.assertTrue(_secret_key(key))
                for shape, text in shapes(key, SECRET).items():
                    self.assertEqual([SECRET], found(text), shape)

    def test_a_one_time_password_is_hidden_in_a_sentence(self):
        for key in ("OTP", "otp", "TOTP", "db_otp", "OTP_CODE", "otpCode"):
            with self.subTest(key=key):
                said = redact_value(f"Das {key} lautet {SECRET}.")
                self.assertNotIn(SECRET, said)
                self.assertIn(key, said)

    def test_a_word_holding_one_names_nothing(self):
        """Sad path: glued into a longer word the name is no secret's, and
        the value stays -- an opaque one and a plain one alike."""
        for key in OTHER_WORDS:
            for value in (SECRET, "Leinsamen", "4096"):
                with self.subTest(key=key, value=value):
                    self.assertFalse(looks_like_a_credential(value, key))
                    self.assertFalse(_secret_key(key))
                    for shape, text in shapes(key, value).items():
                        self.assertEqual([], found(text), shape)
                    sentence = f"{key}: {value}"
                    self.assertEqual(sentence, redact_value(sentence))

    def test_the_edges_of_a_word(self):
        """Edge cases: a digit after the name is no word's end (as "atan2"
        is none); a hump after an acronym is (WPAPsk). Run together in one
        case a name is lost, as USERPIN is: nothing tells WIFIPSK from
        CROPSK."""
        for key in ("PSK_2", "otp-1", "Seed_", "_dsn"):
            with self.subTest(key=key):
                self.assertTrue(looks_like_a_credential(SECRET, key))
        for key in ("wifipsk", "RANDOMSEED", "sentrydsn", "userotp"):
            with self.subTest(key=key):
                self.assertFalse(_secret_key(key))


if __name__ == "__main__":
    unittest.main()
