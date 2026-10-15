"""A secret under a quoted key, and a password handed to a command.

Masking a review read a Django settings.py through find_secrets and
got nothing back:

    DATABASES = {
        'default': {
            'PASSWORD': 'Kx9vQ2mZ7pL4Wq8rTz',
        }
    }

The same text with double quotes had its password found. So did
config.php's array('password' => '...'), app.py's
os.environ['DB_PASSWORD'] = '...', and 18 of the 30 key shapes of the
review's battery: every rule that reads a key beside its value stopped at
the quote.

- The key reader (_ASSIGNMENT_SHAPE, find_secrets' keyed lines and the
  preview's redact_assignments) took a key of [\\w.\\-\\[\\]] only: no quote.
- The sentence rule (_SECRET_IN_SENTENCE, everything scrub reads) allowed
  [\\w.\\-]{0,20} between the keyword and its ":" or "=", and the quote
  that closes 'PASSWORD' is none of those.
- The member rule (_JSON_MEMBER) read a double-quoted string and ":" only;
  neither a single quote nor "=>" (PHP, Ruby, Perl) was a member.

Each now reads a key in single or double quotes -- 'k', "k" -- and one
subscripted with a quoted name -- ['k'], ["k"], os.environ['k'],
$db['k'], a['b']['c'] --, and "=>" as a separator. A quoted key is judged
by the name inside its quotes, as a JSON member is; a value in quotes after
it is the text between them, as the member rule has it.

And the other half of a review: a password handed to a command on
its command line -- mysql -p..., sshpass -p, redis-cli -a, openssl's
pass:, keytool -storepass, htpasswd -b, requests' auth=(user, pw) and
smtplib's login(user, pw) -- only where the flag or the position names a
password.
"""
import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_secret_spans import left_to_hide, mask  # noqa: E402 -- beside this file

from sift.extract import find_secrets
from sift.extract.base import (_KEYED_LINE, _harmless_setting,
                               redact_assignments, redact_value, scrub)

#: The value every shape below hides: opaque, no word, no vendor's prefix,
#: so only a rule that reads the key beside it can find it.
SECRET = "Kx9vQ2mZ7pL4Wq8rTz"


def found(text: str, filename: str = "") -> "list[str]":
    """The texts find_secrets masks in ``text``, bidi controls aside."""
    return [text[span.start:span.end]
            for span in find_secrets(text, filename) if span.reason != "bidi"]


def masked(text: str, filename: str = "") -> str:
    """``text`` with find_secrets' spans masked, every offset kept."""
    return mask(text, find_secrets(text, filename))


class QuotedKeyShapesTest(unittest.TestCase):
    """Each shape a review found nothing in, in the file it came from:
    find_secrets hides the value and keeps the key, and scrub -- what reads
    a summary, a note, a model's answer -- hides the value as well."""

    def assertHidden(self, text: str, filename: str, key: str) -> None:
        shown = masked(text, filename)
        self.assertNotIn(SECRET, shown, f"find_secrets published it: {shown!r}")
        self.assertIn(key, shown, f"the key was masked: {shown!r}")
        said = redact_value(text)
        self.assertNotIn(SECRET, said, f"scrub published it: {said!r}")
        self.assertIn(key, said)

    def test_a_python_dict_with_a_single_quoted_key(self):
        self.assertHidden(f"DB = {{'password': '{SECRET}'}}\n",
                          "settings.py", "'password'")

    def test_a_python_dict_with_a_single_quoted_capital_key(self):
        self.assertHidden(f"DB = {{'PASSWORD': '{SECRET}'}}\n",
                          "settings.py", "'PASSWORD'")

    def test_a_single_quoted_api_token(self):
        self.assertHidden(f"AUTH = {{'api_token': '{SECRET}'}}\n",
                          "settings.py", "'api_token'")

    def test_a_single_quoted_secret_key(self):
        self.assertHidden(f"CFG = {{'secret_key': '{SECRET}'}}\n",
                          "settings.py", "'secret_key'")

    DJANGO = ("DATABASES = {\n    'default': {\n"
              "        'ENGINE': 'django.db.backends.postgresql',\n"
              "        'NAME': 'app',\n        'USER': 'app',\n"
              f"        'PASSWORD': '{SECRET}',\n"
              "        'HOST': 'db',\n    }\n}\n")

    def test_djangos_databases_setting(self):
        """The finding itself: a settings.py, and the same text with no
        name at all."""
        for name in ("settings.py", ""):
            with self.subTest(name=name):
                self.assertHidden(self.DJANGO, name, "'PASSWORD': '")

    def test_a_php_array_with_an_arrow(self):
        self.assertHidden(f"<?php return ['password' => '{SECRET}'];\n",
                          "config.php", "'password' => '")

    def test_a_php_array_with_an_arrow_and_double_quotes(self):
        self.assertHidden(f'<?php return ["password" => "{SECRET}"];\n',
                          "config.php", '"password" => "')

    def test_a_php_array_call(self):
        self.assertHidden(f"$db = array('password' => '{SECRET}');\n",
                          "config.php", "'password' => '")

    def test_a_javascript_object_with_a_single_quoted_key(self):
        self.assertHidden(f"module.exports = {{ 'password': '{SECRET}' }};\n",
                          "config.js", "'password': '")

    def test_yaml_with_single_quotes(self):
        self.assertHidden(f"'password': '{SECRET}'\n", "config.yml",
                          "'password': '")

    def test_a_ruby_hash_with_an_arrow(self):
        self.assertHidden(f"{{ 'password' => '{SECRET}' }}\n", "config.rb",
                          "'password' => '")

    def test_os_environ_set_with_double_quotes(self):
        self.assertHidden(f'os.environ["DB_PASSWORD"] = "{SECRET}"\n',
                          "app.py", 'os.environ["DB_PASSWORD"] = "')

    def test_os_environ_set_with_single_quotes(self):
        self.assertHidden(f"os.environ['DB_PASSWORD'] = '{SECRET}'\n",
                          "app.py", "os.environ['DB_PASSWORD'] = '")

    def test_a_dict_item_set_with_single_quotes(self):
        self.assertHidden(f"config['password'] = '{SECRET}'\n", "app.py",
                          "config['password'] = '")

    def test_a_dict_item_set_with_double_quotes(self):
        self.assertHidden(f'config["password"] = "{SECRET}"\n', "app.py",
                          'config["password"] = "')

    def test_a_header_set_in_javascript(self):
        self.assertHidden(f"headers['X-Api-Key'] = '{SECRET}';\n", "app.js",
                          "headers['X-Api-Key'] = '")

    def test_a_perl_hash(self):
        self.assertHidden(f"my %db = ( 'password' => '{SECRET}' );\n",
                          "cfg.pl", "'password' => '")

    def test_a_lua_table(self):
        self.assertHidden(f"return {{ ['password'] = '{SECRET}' }}\n",
                          "cfg.lua", "['password'] = '")

    # -- the same rules, shapes the review's battery did not hold ----------
    def test_yaml_with_a_double_quoted_key_and_a_plain_value(self):
        self.assertHidden(f'"password": {SECRET}\n', "config.yml",
                          '"password": ')

    def test_yaml_with_a_single_quoted_key_and_a_plain_value(self):
        self.assertHidden(f"'password': {SECRET}\n", "config.yml",
                          "'password': ")

    def test_flasks_secret_key(self):
        self.assertHidden(f"app.config['SECRET_KEY'] = '{SECRET}'\n",
                          "app.py", "app.config['SECRET_KEY'] = '")

    def test_a_php_array_item(self):
        self.assertHidden(f"$db['password'] = '{SECRET}';\n", "config.php",
                          "$db['password'] = '")

    def test_nested_subscripts(self):
        self.assertHidden(
            f"DATABASES['default']['PASSWORD'] = '{SECRET}'\n", "settings.py",
            "DATABASES['default']['PASSWORD'] = '")

    def test_a_ruby_symbol_with_an_arrow(self):
        self.assertHidden(f"{{ :password => '{SECRET}' }}\n", "config.rb",
                          ":password => ")

    def test_a_perl_bareword_with_an_arrow(self):
        self.assertHidden(f"my %db = ( password => '{SECRET}' );\n", "cfg.pl",
                          "password => ")


class QuotedKeyExtentTest(unittest.TestCase):
    """What is masked: the text between a quoted value's quotes, as a JSON
    member's -- so a pretty-printed JSON file keeps its quotes and commas,
    and the preview hides no more than find_secrets masks."""

    def test_a_quoted_value_is_masked_between_its_quotes(self):
        for text in (f"        'PASSWORD': '{SECRET}',\n",
                     f'  "password": "{SECRET}",\n',
                     f"os.environ['DB_PASSWORD'] = '{SECRET}'\n",
                     f"$db = array('password' => '{SECRET}');\n",
                     f'{{"password": "{SECRET}"}}'):
            with self.subTest(text=text):
                self.assertEqual([SECRET], found(text))

    def test_a_pretty_printed_json_file_is_masked_as_before(self):
        """The member rule masked the string's text; the key reader, which
        now reads each of these lines too, masks the same."""
        text = json.dumps({"host": "db", "password": SECRET, "port": 5432,
                           "token": None, "api_key": ""}, indent=2)
        self.assertEqual([SECRET], found(text, "config.json"))
        json.loads(masked(text, "config.json"))

    def test_the_preview_hides_what_find_secrets_masks(self):
        """redact_assignments reads the same key and the same value: after
        the mask, it finds nothing left to hide, and it hides no quote."""
        body = QuotedKeyShapesTest.DJANGO
        self.assertEqual([], left_to_hide(body, masked(body, "settings.py"),
                                          everything=False))
        preview = redact_assignments(body)
        self.assertNotIn(SECRET, preview)
        self.assertIn(f"'PASSWORD': '<{len(SECRET)} chars — not shown>',",
                      preview)
        self.assertIn("'ENGINE': 'django.db.backends.postgresql',", preview)


class QuotedKeyPrecisionTest(unittest.TestCase):
    """A quoted key that names no secret, a value that is none, and prose
    that quotes a word: nothing is masked."""

    def test_a_service_accounts_type(self):
        for text in ("{'type': 'service_account', 'project_id': 'app-1'}\n",
                     "'type' => 'service_account',\n",
                     "config['type'] = 'service_account'\n"):
            with self.subTest(text=text):
                self.assertEqual([], found(text, "app.py"))
                self.assertEqual(text, redact_value(text))

    def test_the_other_settings_of_djangos_databases(self):
        self.assertEqual([SECRET],
                         found(QuotedKeyShapesTest.DJANGO, "settings.py"))

    def test_a_key_about_a_secret_is_exempt_as_a_bare_one_is(self):
        """_names_the_secret_not_the_value reads the name inside the quotes:
        read with them, "password_hash'" ends in no suffix it knows."""
        for key, value in (("password_hash", "pbkdf2_sha256"),
                           ("password_min_length", "12"),
                           ("token_expiry", "3600"),
                           ("max_tokens", "4096"),
                           ("api_key_env", "SIFT_API_KEY")):
            for text in (f"'{key}': '{value}'\n", f'"{key}" => "{value}"\n',
                         f"os.environ['{key.upper()}'] = '{value}'\n"):
                with self.subTest(text=text):
                    self.assertEqual([], found(text, "app.py"))
                    self.assertEqual(text, redact_value(text))

    def test_an_empty_or_absent_password(self):
        for text in ("'password': '',\n", "'password': None,\n",
                     '  "password": null,\n', "'password' => '',\n",
                     "'password': true\n"):
            with self.subTest(text=text):
                self.assertEqual([], found(text, "app.py"))

    def test_a_quoted_key_naming_nothing(self):
        for text in ("'user' => 'admin',\n", "'host': 'db.intern',\n",
                     "headers['Content-Type'] = 'application/json'\n",
                     "os.environ['LANG'] = 'C.UTF-8'\n"):
            with self.subTest(text=text):
                self.assertEqual([], found(text, "app.py"))
                self.assertEqual(text, redact_value(text))

    def test_prose_that_quotes_the_word(self):
        for text in ("Enter the 'password' field and press Enter.",
                     'The key "password" is required: see the manual.',
                     "The \"token\" = a string the server hands out.",
                     "Type 'password' => then press Tab."):
            with self.subTest(text=text):
                self.assertEqual([], found(text))
                self.assertEqual(text, redact_value(text))

    def test_an_arrow_inside_a_string(self):
        """"=>" inside a quoted value or key is the string's, not a
        separator."""
        for text in ("'note' => 'use => for arrays',\n",
                     "log.info('password => see the vault')\n",
                     "'a => b': 'password'\n"):
            with self.subTest(text=text):
                self.assertEqual([], found(text, "app.php"))
                self.assertEqual(text, redact_value(text))


    def test_code_after_a_quoted_key_is_no_value(self):
        """An expression of the code the key is in -- a name, a call, a
        subscript, a collection, a number -- is not the password; a literal
        in quotes is (_a_literal). The ansible collections in
        /usr/lib/python3 had thousands of these masked."""
        for text in ("        'password': password,\n",
                     "    'PASSWORD': os.environ['DB_PASSWORD'],\n",
                     "    'password' => env('DB_PASSWORD', ''),\n",
                     '    "access_token": {"required": False, "no_log": True},\n',
                     "    'password': KeyInfo(),\n",
                     "    'pass_stmt': tree.KeywordStatement,\n",
                     "opts['forcepasswordchange'] = self.forcepasswordchange\n",
                     "    'token': $token,\n",
                     '    "key_tag": 18948,\n',
                     "    'pin': -1,\n"):
            with self.subTest(text=text):
                self.assertEqual([], found(text, "module.py"))
                self.assertEqual(text, redact_value(text))
                self.assertEqual(text.rstrip("\n"), redact_assignments(text))

    def test_html_attributes_are_no_quoted_keys(self):
        """A quoted key is joined to its value by ":", "=" or "=>": an
        attribute's value in quotes, a blank and the next attribute are not
        -- "pin" in "mapping" and "grouping" is no key's word either."""
        for text in ('<a class="headerlink" href="#register-mapping" '
                     'title="Link to this heading">',
                     '<property name="grouping" type="uint" value="1"/>',
                     '<link rel="prev" title="CTEST_DROP_SITE_PASSWORD" '
                     'href="CTEST_DROP_SITE_PASSWORD.html" />',
                     'Search.setIndex({"alltitles":{"Mapping":[[3,"id117"]],'
                     '"Stepping":[[4,"id5"]]}})'):
            with self.subTest(text=text):
                self.assertEqual([], found(text, "page.html"))
                self.assertEqual(text, redact_value(text))

    def test_json_in_a_single_quoted_string_is_still_a_member(self):
        """The members of a key in single quotes are read in a scan of
        their own: an apostrophe, or a Python string around the JSON, opens
        a string to that scan, and does not hide the JSON's members from
        the scan that reads double quotes."""
        for text in (f"payload = '{{\"auth\": \"{SECRET}\"}}'\n",
                     f"Don't commit {{\"auth\": \"{SECRET}\"}} again.\n",
                     f"it's {{'auth': '{SECRET}'}} and {{\"auth\": \"{SECRET}\"}}\n"):
            with self.subTest(text=text):
                self.assertNotIn(SECRET, masked(text, "app.py"))

class QuotedKeyEdgeTest(unittest.TestCase):

    def test_a_long_quoted_key(self):
        """A key of 1,024 characters, the key reader's longest (a host application's
        KEY_REACH is held to it), quotes included; one longer is no key to
        the key reader, and the member rule still reads it."""
        name = "x" * (1024 - len("''_password")) + "_password"
        line = f"'{name}': '{SECRET}'"
        self.assertIsNotNone(_KEYED_LINE.match(line))
        self.assertEqual([SECRET], found(line + "\n"))
        longer = f"'x{name}': '{SECRET}'"
        self.assertIsNone(_KEYED_LINE.match(longer))
        self.assertEqual([SECRET], found(longer + "\n"))

    def test_nested_brackets(self):
        for text in (f"config['db']['password'] = '{SECRET}'\n",
                     f"cfg[0]['password'] = '{SECRET}'\n",
                     f"cfg['a[1]']['password'] = '{SECRET}'\n"):
            with self.subTest(text=text):
                self.assertEqual([SECRET], found(text, "app.py"))

    def test_an_escaped_quote_inside_the_key(self):
        text = f"{{'user\\'s password': '{SECRET}'}}\n"
        self.assertEqual([SECRET], found(text, "app.py"))
        self.assertNotIn(SECRET, redact_value(text))
        text = "{'label\\'s': 'password', 'x': 'y'}\n"
        self.assertEqual([], found(text, "app.py"))

    def test_a_value_with_blanks_between_its_quotes(self):
        """Between its quotes a value runs over blanks: the sentence rule
        ends a bare value at the first."""
        text = "'password': 'Kx9v Q2mZ7pL4Wq8rTz',\n"
        self.assertEqual(["Kx9v Q2mZ7pL4Wq8rTz"], found(text, "app.py"))
        self.assertNotIn("Q2mZ7pL4Wq8rTz", redact_value(text))

    def test_a_passphrase_of_words_under_a_quoted_key(self):
        """The key names a password: the words are the password, as under a
        bare key on its own line."""
        text = "'password': 'correct horse battery staple',\n"
        self.assertEqual(["correct horse battery staple"],
                         found(text, "app.py"))


class KeyGroupTest(unittest.TestCase):
    """_KEYED_LINE's groups, which a host application's viewer reads (_key_line: the
    key in group 1, the value from match.start(3)): the key as the line
    writes it, blanks and quotes included; the separator; the rest."""

    def test_the_groups_of_a_quoted_key(self):
        cases = {
            f"        'PASSWORD': '{SECRET}',":
                ("        'PASSWORD'", ":", f"'{SECRET}',"),
            f'"password": {SECRET}': ('"password"', ":", SECRET),
            f"os.environ['DB_PASSWORD'] = '{SECRET}'":
                ("os.environ['DB_PASSWORD']", "=", f"'{SECRET}'"),
            f"  'password' => '{SECRET}',":
                ("  'password'", "=>", f"'{SECRET}',"),
            "export ['k'] = v": ("export ['k']", "=", "v"),
        }
        for line, groups in cases.items():
            with self.subTest(line=line):
                self.assertEqual(groups, _KEYED_LINE.match(line).groups())

    def test_the_groups_of_a_bare_key_are_as_before(self):
        """0.1.74's pattern, on lines with no quote before the separator
        and no "=>": the same three groups."""
        old = re.compile(r"^([ \t]*(?:export[ \t]+)?[\w.\-\[\]]{1,1024})"
                         r"[ \t]*([:=])[ \t]*(\S.*)$")
        lines = ["KEY=value", "  key: value", "export A=b # c", "k.v[0]: y",
                 "a=b=c", "a: 'b': c", "x = \"y\"", "a[b] = 'c'", "k:",
                 "cfg := Config{", "k =", "k==v", "k:v:w", "\tk\t=\tv"]
        for line in lines:
            with self.subTest(line=line):
                new = _KEYED_LINE.match(line)
                before = old.match(line)
                self.assertEqual(before and before.groups(),
                                 new and new.groups())

    def test_a_quoted_key_is_judged_by_the_name_in_its_quotes(self):
        """A host application hands the key as group 1 has it, quotes included, to
        _harmless_setting: every quoted and subscripted spelling of a key is
        judged as the key is."""
        keys = ["PASSWORD", "DB_PASSWORD", "LOG_LEVEL", "PORT", "ALARM_CODE",
                "ZIP_CODE", "masterKey", "api_key_env", "MONKEY", "TZ"]
        values = ["true", "info", "Kx9vQ2mZ7pL4", "5432", "4711", "",
                  "${X}", "/var/log", "UTF-8", "sonnenschein", "hunter2"]
        for key in keys:
            for spelled in (f"'{key}'", f'"{key}"', f"['{key}']",
                            f'["{key}"]', f"os.environ['{key}']"):
                with self.subTest(key=spelled):
                    self.assertEqual(
                        [_harmless_setting(key, v) for v in values],
                        [_harmless_setting(spelled, v) for v in values])

    def test_a_settings_files_quoted_key(self):
        """In a .env every value is a setting, a quoted key's too; its key
        is the one the line writes."""
        from sift.extract.base import _setting_keys
        text = "'LOG_LEVEL'=info\n\"DB_PASSWORD\"=hunter2\n"
        spans = find_secrets(text, ".env")
        self.assertEqual(["info", "hunter2"],
                         [text[s.start:s.end] for s in spans])
        self.assertEqual({text.index("info"): "'LOG_LEVEL'",
                          text.index("hunter2"): '"DB_PASSWORD"'},
                         _setting_keys(text, {s.start for s in spans}))


if __name__ == "__main__":
    unittest.main()


class CommandLineTest(unittest.TestCase):
    """A password handed to a command on its command line: only where the flag or the position names one -- mysql's
    -p<password>, sshpass -p, redis-cli -a, openssl's pass:, keytool's
    -storepass and -keypass, htpasswd -b's last argument, the second of
    requests' auth=(user, password) and of a .login(user, password)."""

    def assertHidden(self, text: str, filename: str = "run.sh") -> None:
        shown = masked(text, filename)
        self.assertNotIn(SECRET, shown, f"find_secrets published it: {shown!r}")
        said = redact_value(text)
        self.assertNotIn(SECRET, said, f"scrub published it: {said!r}")

    def assertKept(self, text: str, filename: str = "run.sh") -> None:
        self.assertEqual([], found(text, filename))
        self.assertEqual(text, redact_value(text))

    # -- happy -------------------------------------------------------------
    def test_mysql_with_the_password_glued_to_p(self):
        for text in (f"mysqldump -u root -p{SECRET} app > app.sql\n",
                     f"mysql -uroot -p'{SECRET}' -e 'SELECT 1'\n",
                     f'mariadb -h db -u app -p"{SECRET}" app\n',
                     f"mysqladmin --host=db -p{SECRET} status\n"):
            with self.subTest(text=text):
                self.assertHidden(text)

    def test_sshpass(self):
        for text in (f"sshpass -p '{SECRET}' ssh deploy@host\n",
                     f"sshpass -p {SECRET} scp app.tar deploy@host:\n",
                     f"sshpass -p{SECRET} ssh host uptime\n",
                     f'sshpass -v -p "{SECRET}" rsync -a . host:/srv\n'):
            with self.subTest(text=text):
                self.assertHidden(text)

    def test_redis_cli(self):
        for text in (f"redis-cli -h cache -a {SECRET} ping\n",
                     f"redis-cli -h cache -p 6379 -a '{SECRET}' info\n",
                     f"redis-cli --user app --pass {SECRET} ping\n"):
            with self.subTest(text=text):
                self.assertHidden(text)

    def test_openssls_pass_argument(self):
        for text in (
                f"openssl pkcs12 -export -passout pass:{SECRET} -out c.p12\n",
                f"openssl rsa -in k.pem -passin 'pass:{SECRET}' -out k2\n",
                f"openssl enc -aes-256-cbc -pass pass:{SECRET} -in a\n"):
            with self.subTest(text=text):
                self.assertHidden(text, "make.sh")

    def test_keytool_and_jarsigner(self):
        for text in (f"keytool -importcert -storepass {SECRET} -file ca.pem\n",
                     f"keytool -genkeypair -alias app -keypass '{SECRET}'\n",
                     f"keytool -importkeystore -srcstorepass {SECRET} "
                     "-destkeystore b.jks\n",
                     f"jarsigner -storepass {SECRET} app.jar alias\n"):
            with self.subTest(text=text):
                self.assertHidden(text, "make.sh")

    def test_htpasswd_in_batch_mode(self):
        for text in (f"htpasswd -b /etc/nginx/.htpasswd admin {SECRET}\n",
                     f"htpasswd -cb .htpasswd admin '{SECRET}'\n",
                     f"htpasswd -nbB admin {SECRET}\n",
                     f"htpasswd -bBC 10 /etc/.htpasswd admin {SECRET}\n",
                     f"htpasswd -B -b -C 12 users admin {SECRET}\n"):
            with self.subTest(text=text):
                self.assertHidden(text, "setup.sh")

    def test_requests_auth_tuple(self):
        for text in (f"requests.get(url, auth=('admin', '{SECRET}'))\n",
                     f'httpx.get(url, auth=(user, "{SECRET}"))\n',
                     f"session.auth = ('admin', '{SECRET}')\n"):
            with self.subTest(text=text):
                self.assertHidden(text, "client.py")

    def test_a_login_call(self):
        for text in (f"server.login('ops@example.org', '{SECRET}')\n",
                     f"imap.login(USER, '{SECRET}')\n",
                     f'ftp.login("backup", "{SECRET}")\n'):
            with self.subTest(text=text):
                self.assertHidden(text, "mail.py")

    # -- sad ---------------------------------------------------------------
    def test_a_flag_that_names_no_password(self):
        for text in ("mysql -u root -p app < dump.sql\n",
                     "mysql -h db -P 3306 -u app app\n",
                     "gcc -pthread -o x x.c\n", "ssh -p2222 host\n",
                     "mkdir -p /tmp/x && tar -pcf a.tar dir\n",
                     "grep -a pattern file | redis-cli -h cache ping\n",
                     "ls -a; keytool -list -keystore ks.jks\n",
                     "htpasswd -c /etc/nginx/.htpasswd admin\n",
                     "htpasswd -D /etc/nginx/.htpasswd admin\n",
                     "htpasswd -v /etc/nginx/.htpasswd admin\n"):
            with self.subTest(text=text):
                self.assertKept(text)

    def test_a_password_read_from_elsewhere(self):
        for text in ('mysqldump -u root -p"$MYSQL_PWD" app\n',
                     "mysql -uroot -p${DB_PASSWORD} app\n",
                     'sshpass -p "$SSHPASS" ssh host\n',
                     'sshpass -p "$(cat ~/.pw)" ssh host\n',
                     "sshpass -e ssh host\n", "sshpass -f ~/.pw ssh host\n",
                     "redis-cli -a $REDIS_PASSWORD ping\n",
                     "openssl pkcs12 -export -passout env:P12_PASS -out c\n",
                     "openssl rsa -passin file:pw.txt -in k.pem\n",
                     "openssl rsa -passin stdin -in k.pem\n",
                     "keytool -list -storepass:env STOREPASS\n",
                     "keytool -list -storepass:file pw.txt\n",
                     "htpasswd -b users admin \"$ADMIN_PW\"\n",
                     "cmd = 'mysql -u %s -p%s' % (user, pw)\n"):
            with self.subTest(text=text):
                self.assertKept(text)

    def test_code_that_passes_a_variable(self):
        for text in ("requests.get(url, auth=(user, password))\n",
                     "requests.get(url, auth=('admin', os.environ['PW']))\n",
                     "requests.get(url, auth=HTTPBasicAuth(u, p))\n",
                     "server.login(user, password)\n", "ftp.login()\n",
                     "smtp.login(self.username, self.password)\n",
                     "ftp.login('anonymous', '')\n"):
            with self.subTest(text=text):
                self.assertKept(text, "client.py")

    def test_a_synopsis_names_the_argument(self):
        """A manual's synopsis writes the argument's name where the value
        goes -- -ppassword, -storepass storepass, [:env | :file], troff's
        \\fIkeypass\\fR -- and a name for the secret is no secret."""
        for text in ("sshpass [-ffilename|-dnum|-ppassword|-e] command\n",
                     "mysql [-p[password]] db\n",
                     "keytool [-storepass storepass] [-keypass keypass]\n",
                     "keytool [-keypass old_keypass] [-new new_keypass]\n",
                     "jarsigner -keypass [:env | :file] argument\n",
                     "\\f3[\\-storepass \\fIstorepass\\fR]\\fP\n",
                     "htpasswd -b passwdfile username password\n",
                     "htpasswd -nb username password\n",
                     "redis-cli -a <password> ping\n"):
            with self.subTest(text=text):
                self.assertKept(text, "tool.1")

    # -- edge --------------------------------------------------------------
    def test_a_command_continued_over_lines(self):
        text = f"mysqldump \\\n  -u root \\\n  -p{SECRET} \\\n  app > app.sql\n"
        self.assertHidden(text)

    def test_a_quoted_password_with_blanks(self):
        """Between its quotes, blanks and all; the quotes stay, as a quoted
        key's value's do."""
        text = f"sshpass -p 'Kx9v {SECRET}' ssh host\n"
        self.assertEqual([f"Kx9v {SECRET}"], found(text))

    def test_the_password_alone_is_masked(self):
        """The flag, the user and the rest of the command stay readable."""
        cases = {f"htpasswd -b /etc/.htpasswd admin {SECRET}\n":
                     "htpasswd -b /etc/.htpasswd admin ",
                 f"mysqldump -u root -p{SECRET} app\n": "mysqldump -u root -p",
                 f"requests.get(url, auth=('admin', '{SECRET}'))\n":
                     "requests.get(url, auth=('admin', '"}
        for text, kept in cases.items():
            with self.subTest(text=text):
                self.assertEqual([SECRET], found(text))
                self.assertTrue(masked(text).startswith(kept))
