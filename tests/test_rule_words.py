"""The words of the secret rules at three edges.

A rule word ending a sentence took the next sentence's key for its value:
"reveal the secrets. password: Kx9v..." found nothing while "the team.
password: Kx9v..." found the value, since the sentence rule's key run
crossed the "." that ended the sentence and the scan resumed past it.

A name of two English words written with a blank -- "api key", "secret
key", "master key" -- named nothing to the sentence rule and the key tiers
while api_key did.

A key word glued to "key" -- MASTERKEY, masterKey, SigningKey -- named a
secret to the settings rule alone; an assignment line and a sentence read
it as no key. A word merely ending in "key" -- monkey, hockey -- names
none, in any rule.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sift.extract import find_secrets
from sift.extract.base import (_glued_key, _secret_key,
                               looks_like_a_credential, redact_value)

#: A value no rule hides for its own shape: only a key naming one can.
VALUE = "Kx9vQ2mZ" + "7pL4Wq8rTz"


def found(text: str, name: str = "") -> "list[str]":
    """The texts find_secrets masks in ``text``, bidi controls aside."""
    return [text[span.start:span.end]
            for span in find_secrets(text, name) if span.reason != "bidi"]


class RuleWordEndsASentenceTest(unittest.TestCase):

    ENDINGS = ("reveal the secrets.", "rotate the token.", "set a password.",
               "check the credentials.", "the api key.")

    def test_a_value_after_a_sentence_ending_in_a_rule_word(self):
        for ending in self.ENDINGS:
            text = f"{ending} password: {VALUE}"
            with self.subTest(ending=ending):
                self.assertEqual([VALUE], found(text))
                self.assertNotIn(VALUE, redact_value(text))

    def test_the_control_with_no_rule_word_before_it(self):
        self.assertEqual([VALUE], found(f"the team. password: {VALUE}"))

    # -- sad ---------------------------------------------------------------
    def test_other_punctuation_was_never_crossed(self):
        for ending in ("the secrets;", "the secrets!", "the secrets -"):
            with self.subTest(ending=ending):
                self.assertEqual([VALUE], found(f"{ending} password: {VALUE}"))

    def test_a_sentence_ending_in_a_rule_word_with_no_value_after_it(self):
        for text in ("reveal the secrets. The team met.",
                     "Set a password. Then log in.",
                     "rotate the token. password policy: 12 characters"):
            with self.subTest(text=text):
                self.assertEqual([], found(text))
                self.assertEqual(text, redact_value(text))

    # -- edge --------------------------------------------------------------
    def test_a_dot_inside_a_key_is_the_keys(self):
        for key in ("v1.2.password", "password.v2", "db.password.prod"):
            with self.subTest(key=key):
                self.assertEqual([VALUE], found(f"{key}: {VALUE}"))

    def test_a_dot_ending_the_text(self):
        self.assertEqual([], found("reveal the secrets."))
        self.assertNotIn(VALUE, redact_value(f"password: {VALUE}."))


class TwoWordNamesTest(unittest.TestCase):

    NAMES = ("api key", "API Key", "secret key", "private key", "access key",
             "master key", "access token", "auth token", "client secret",
             "encryption key", "signing key")

    def test_the_name_with_a_blank_in_a_sentence(self):
        for name in self.NAMES:
            for text in (f"Subject: {name} {VALUE}", f"{name}: {VALUE}",
                         f"The {name} is {VALUE}."):
                with self.subTest(text=text):
                    self.assertEqual([VALUE], found(text))

    def test_the_name_with_a_blank_as_a_key(self):
        """A CSV column, a quoted key and a JSON member write a blank."""
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertTrue(looks_like_a_credential(VALUE, name))
                self.assertTrue(_secret_key(name))
                self.assertEqual([VALUE], found(f'"{name}": "{VALUE}"'))
                self.assertEqual([VALUE], found(f"'{name}': '{VALUE}'"))

    def test_the_name_with_a_separator_as_before(self):
        for name in ("api_key", "api-key", "api.key", "apikey", "API_KEY"):
            with self.subTest(name=name):
                self.assertEqual([VALUE], found(f"Subject: {name} {VALUE}"))

    # -- sad ---------------------------------------------------------------
    def test_prose_about_the_name_stays_prose(self):
        for text in ("api keys are documented here",
                     "The api key is stored in the vault.",
                     "The API Key rotation is due in 90 days.",
                     "the secret key file is /etc/app/signing.pem",
                     "private key: /etc/ssl/private/server.key",
                     "an access token expires after 3600 seconds"):
            with self.subTest(text=text):
                self.assertEqual([], found(text))
                self.assertEqual(text, redact_value(text))

    def test_two_blanks_are_no_name(self):
        self.assertEqual([], found(f"api  key {VALUE}"))

    def test_a_word_before_a_json_strings_close_is_a_word(self):
        """An LLM server's error body, as sift's own message quotes it
        (test_llm_clients): the closing quote and bracket after a word say
        nothing about it."""
        for text in ('HTTP 403 at /models: {"error": "api key invalid"}',
                     '{"error": "password invalid"}', "{'detail': 'token expired'}",
                     "(the secret key is missing)"):
            with self.subTest(text=text):
                self.assertEqual(text, redact_value(text))

    def test_a_credential_before_a_json_strings_close_is_hidden(self):
        for text in ('{"note": "password: hunter2"}', f"(api key {VALUE})"):
            with self.subTest(text=text):
                self.assertNotIn("hunter2", redact_value(text))
                self.assertNotIn(VALUE, redact_value(text))

    def test_a_blank_before_a_suffix_about_the_secret(self):
        for name in ("api key name", "Token Expiry", "Password Hash",
                     "max tokens"):
            with self.subTest(name=name):
                self.assertFalse(looks_like_a_credential(VALUE, name))

    # -- edge --------------------------------------------------------------
    def test_a_plural_and_a_label(self):
        self.assertEqual([VALUE], found(f"api keys: {VALUE}"))
        self.assertEqual([VALUE], found(f"API Key 2: {VALUE}"))


class GluedKeyTest(unittest.TestCase):

    GLUED = ("MASTERKEY", "masterKey", "MasterKey", "masterkey", "APIKey",
             "SigningKey", "encryptionKey", "AUTHKEY", "sessionKey",
             "LICENSEKEY", "db.masterKey", "masterKeys")

    def test_a_glued_key_names_a_key(self):
        for key in self.GLUED:
            with self.subTest(key=key):
                self.assertTrue(_glued_key(key))
                self.assertTrue(_secret_key(key))
                self.assertTrue(looks_like_a_credential(VALUE, key))
                for name in ("a.yaml", "a.txt"):
                    self.assertEqual([VALUE], found(f"{key}: {VALUE}\n", name))
                self.assertEqual([VALUE], found(f'"{key}": "{VALUE}"'))

    def test_a_glued_key_in_a_sentence(self):
        for key in ("MASTERKEY", "masterKey", "SigningKey"):
            for text in (f"the {key}: {VALUE}", f"the {key} is {VALUE}"):
                with self.subTest(text=text):
                    self.assertEqual([VALUE], found(text))

    def test_a_glued_key_in_a_dotenv_as_before(self):
        self.assertEqual([VALUE], found(f"MASTERKEY={VALUE}\n", ".env"))

    # -- sad ---------------------------------------------------------------
    def test_a_word_ending_in_key_names_none(self):
        for key in ("monkey", "turkey", "hockey", "donkey", "hotkey",
                    "turnkey", "MONKEY", "hockeyTeam"):
            for value in ("banana", "yes", "3", VALUE, "Ctrl+Shift+K"):
                with self.subTest(key=key, value=value):
                    self.assertFalse(_glued_key(key))
                    self.assertFalse(looks_like_a_credential(value, key))
                    text = f"{key}: {value}"
                    self.assertEqual([], found(text, "a.yaml"))
                    self.assertEqual(text, redact_value(text))

    def test_a_glued_key_hints_only(self):
        """As SECRETISH's "key" does: the value must look like one."""
        for value in ("primary", "true", "${MASTER_KEY}", "none"):
            with self.subTest(value=value):
                self.assertEqual([], found(f"masterKey: {value}\n", "a.yaml"))

    # -- edge --------------------------------------------------------------
    def test_a_glued_key_about_the_secret(self):
        """A path under such a key is a reference, as under apiKeyFile."""
        for key in ("masterKeyFile", "MASTERKEY_PATH", "signingKeyName"):
            with self.subTest(key=key):
                self.assertFalse(looks_like_a_credential("/etc/k.pem", key))
        self.assertFalse(looks_like_a_credential(VALUE, "MASTERKEY_PATH"))


if __name__ == "__main__":
    unittest.main()
