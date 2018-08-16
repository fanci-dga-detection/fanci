from unittest import TestCase

from data_processing.data import LabeledData
from learning import feature_extraction

from learning.feature_extraction import extract_features, _vowel_ratio, _length, _parts, _digit_ratio, \
    _contains_ipv4_addr, _contains_digits, _has_valid_tld, _contains_one_char_subdomains, \
    _contains_wwwdot, _subdomain_lengths_mean, _prefix_repetition, _char_diversity, _contains_subdomain_of_only_digits, \
    _contains_tld_as_infix, _n_grams, _hex_part_ratio, _underscore_ratio, _alphabet_size, _shannon_entropy, \
    _ratio_of_repeated_chars, _consecutive_consonant_ratio, _consecutive_digits_ratio

rwth = 'itsec.rwth-aachen.de'
digits = '123huhu384gj8nge8.co.uk'
subdomains = '1.2.3.4.5.6.7.8.9.123huhu384gj8nge8.co.uk'
vowel_one = 'eeeeeeeeeeeeeee'
vowel_zero = 'ggggg123ggghjkl12323.de'
mhm = 'm.hm'
one_digit = '1'
one_letter = 'a'
only_tld = 'de'
double_tld = 'de.de'
triple_tld = 'de.de.de'
ip_addr = 'hu.10.13.37.19.rwth-aachen.de'
wwwdot = 'wegonweg.www.egioegn.de'
wwwtrailingdot = 'www.egioegn.de'
prefix_repeat = 'rwth-aachen.derwth-aachen.de'
ngrameasy = 'abcdef.de'
underscores = '_tcp.ab_cdef1.de'


class TestExtractFeatures(TestCase):

    def setUp(self):
        self.feature_list = _length, _parts, _vowel_ratio, _digit_ratio, _contains_ipv4_addr, _contains_digits, \
                            _has_valid_tld, _contains_one_char_subdomains, _contains_wwwdot, _subdomain_lengths_mean, \
                            _prefix_repetition, _char_diversity, _contains_subdomain_of_only_digits, _contains_tld_as_infix, \
                            _n_grams, _hex_part_ratio, _underscore_ratio, _alphabet_size, _shannon_entropy, \
                            _ratio_of_repeated_chars, _consecutive_consonant_ratio, _consecutive_digits_ratio
        self.rwth = feature_extraction.extract_features(rwth, self.feature_list)
        self.digits = feature_extraction.extract_features(digits, self.feature_list)
        self.subdomains = feature_extraction.extract_features(subdomains, self.feature_list)
        self.vowel_one = feature_extraction.extract_features(vowel_one, self.feature_list)
        self.vowel_zero = feature_extraction.extract_features(vowel_zero, self.feature_list)
        self.mhm = feature_extraction.extract_features(mhm, self.feature_list)
        self.one_digit = feature_extraction.extract_features(one_digit, self.feature_list)
        self.one_letter = feature_extraction.extract_features(one_letter, self.feature_list)
        self.only_tld = feature_extraction.extract_features(only_tld, self.feature_list)
        self.double_tld = feature_extraction.extract_features(double_tld, self.feature_list)
        self.triple_tld = feature_extraction.extract_features(triple_tld, self.feature_list)
        self.ip_addr = feature_extraction.extract_features(ip_addr, self.feature_list)
        self.wwwdot = feature_extraction.extract_features(wwwdot, self.feature_list)
        self.wwwtrailingdot = feature_extraction.extract_features(wwwtrailingdot, self.feature_list)
        self.prefix_repeat = feature_extraction.extract_features(prefix_repeat, self.feature_list)
        self.ngrameasy = feature_extraction.extract_features(ngrameasy, self.feature_list)
        self.underscores = feature_extraction.extract_features(underscores, self.feature_list)

    def test_length(self):
        # pos 0
        self.assertEqual(self.rwth[0], len(rwth))
        self.assertEqual(self.digits[0], len(digits))

    def test_parts(self):
        # pos 1 to 4

        self.assertEqual(self.rwth[1], 0)
        self.assertEqual(self.rwth[2], 1)
        self.assertEqual(self.rwth[3], 0)
        self.assertEqual(self.rwth[4], 0)

        self.assertEqual(self.digits[1], 1)
        self.assertEqual(self.digits[2], 0)
        self.assertEqual(self.digits[3], 0)
        self.assertEqual(self.digits[4], 0)

        self.assertEqual(self.subdomains[1], 0)
        self.assertEqual(self.subdomains[2], 0)
        self.assertEqual(self.subdomains[3], 0)
        self.assertEqual(self.subdomains[4], 1)

    def test_vowel_ratio(self):
        # pos 5
        self.assertEqual(self.rwth[5], 5 / 15)
        self.assertEqual(self.digits[5], 3 / 9)
        self.assertEqual(self.subdomains[5], 3 / 9)
        self.assertEqual(self.vowel_one[5], 1)
        self.assertEqual(self.vowel_zero[5], 0)

    def test_digit_ratio(self):
        # pos 6
        self.assertEqual(self.digits[6], 8 / 17)

    def test_contains_ip_addr(self):
        # pos 7
        self.assertEqual(self.rwth[7], 0)
        self.assertEqual(self.one_letter[7], 0)
        self.assertEqual(self.ip_addr[7], 1)
        self.assertEqual(self.one_digit[7], 0)
        self.assertEqual(self.mhm[7], 0)

    def test_contains_digits(self):
        # pos 8
        self.assertEqual(self.ip_addr[8], 1)
        self.assertEqual(self.one_digit[8], 1)
        self.assertEqual(self.one_letter[8], 0)
        self.assertEqual(self.rwth[8], 0)

    def test_has_valid_tld(self):
        # pos 9
        self.assertEqual(self.rwth[9], 1)
        self.assertEqual(self.one_digit[9], 0)
        self.assertEqual(self.only_tld[9], 0)
        self.assertEqual(self.vowel_zero[9], 1)
        self.assertEqual(self.vowel_one[9], 0)

    def test_contains_one_char_subdomains(self):
        # pos 10
        self.assertEqual(self.rwth[10], 0)
        self.assertEqual(self.one_letter[10], 1)
        self.assertEqual(self.one_digit[10], 1)
        self.assertEqual(self.vowel_zero[10], 0)
        self.assertEqual(self.vowel_one[10], 0)
        self.assertEqual(self.subdomains[10], 1)

    def test_contains_wwwdot(self):
        #pos 11
        self.assertEqual(self.subdomains[11], 0)
        self.assertEqual(self.wwwdot[11], 1)
        self.assertEqual(self.wwwtrailingdot[11], 1)
        self.assertEqual(self.rwth[11], 0)

    def test_subdomain_lengths_mean(self):
        # pos 12
        self.assertEqual(self.rwth[12], 8)
        self.assertEqual(self.subdomains[12], 26 / 10)

    def test_prefix_repetition(self):
        # pos 13
        self.assertEqual(self.rwth[13], 0)
        self.assertEqual(self.subdomains[13], 0)
        self.assertEqual(self.double_tld[13], 0)
        self.assertEqual(self.prefix_repeat[13], 1)

    def test_char_diversity(self):
        # pos 14
        self.assertEqual(self.one_letter[14], 1)
        self.assertEqual(self.one_digit[14], 1)
        self.assertEqual(self.rwth[14], 11 / 16)
        self.assertEqual(self.triple_tld[14], 1 / 2)

    def test_contains_subdomain_of_only_digits(self):
        # pos 15
        self.assertEqual(self.rwth[15], 0)
        self.assertEqual(self.subdomains[15], 1)
        self.assertEqual(self.one_digit[15], 1)
        self.assertEqual(self.double_tld[15], 0)
        self.assertEqual(self.ip_addr[15], 1)
        self.assertEqual(self.prefix_repeat[15], 0)

    def test_contains_tld_as_infix(self):
        # pos 16
        self.assertEqual(self.rwth[16], 0)
        self.assertEqual(self.subdomains[16], 0)
        self.assertEqual(self.one_digit[16], 0)
        self.assertEqual(self.triple_tld[16], 1)
        self.assertEqual(self.prefix_repeat[16], 0)

    def test_n_grams(self):
        # pos 17 to 37
        # stats = [npa.std(), numpy.median(npa), npa.mean(), numpy.min(npa), numpy.max(npa), numpy.percentile(npa, 25), numpy.percentile(npa, 75)]
        for i in range(24,38):
            self.assertEqual(self.mhm[i], -1)
            self.assertEqual(self.one_digit[i], -1)
            self.assertEqual(self.one_letter[i], -1)

        self.assertEqual(self.ngrameasy[17], 0)
        for i in range(18,24):
            self.assertEqual(self.ngrameasy[i], 1)
        self.assertEqual(self.ngrameasy[18], 1)

        self.assertEqual(self.ngrameasy[24], 0)
        for i in range(25,31):
            self.assertEqual(self.ngrameasy[i], 1)

        self.assertEqual(self.ngrameasy[31], 0)
        for i in range(32,38):
            self.assertEqual(self.ngrameasy[i], 1)

    def test_hex_part_ratio(self):
        # pos 38
        self.assertEqual(self.ngrameasy[38], 1)
        self.assertEqual(self.subdomains[38], 9 / 10)
        self.assertEqual(self.rwth[38], 0)

    def test_underscore_ratio(self):
        # pos 39
        self.assertEqual(self.rwth[39], 0)
        self.assertEqual(self.subdomains[39], 0)
        self.assertEqual(self.prefix_repeat[39], 0)
        self.assertEqual(self.underscores[39], 2 / 12)

    def test_alphabet_size(self):
        # pos 40
        self.assertEqual(self.rwth[40], 11)
        self.assertEqual(self.vowel_one[40], 1)
        self.assertEqual(self.ngrameasy[40], 6)
        self.assertEqual(self.one_digit[40], 1)
        self.assertEqual(self.one_letter[40], 1)

    def test_shannon_entropy(self):
        # pos 41
        self.assertAlmostEqual(self.rwth[41], 3.375, delta=0.1)
        self.assertAlmostEqual(self.vowel_one[41], 0, delta=0.1)
        self.assertAlmostEqual(self.digits[41], 3.33718, delta=0.05)

    def test_ratio_of_repeated_chars(self):
        # pos 42
        self.assertEqual(self.vowel_one[42], 1)
        self.assertEqual(self.one_letter[42], 0)
        self.assertEqual(self.one_digit[42], 0)
        self.assertEqual(self.rwth[42], 5 / 11)

    def test_consecutive_consonant_ratio(self):
        # pos 43
        self.assertEqual(self.one_digit[43], 0)
        self.assertEqual(self.vowel_one[43], 0)
        self.assertEqual(self.rwth[43], 8 / 16)
        self.assertEqual(self.underscores[43], 5 / 12)

    def test_consecutive_digits_ratio(self):
        # pos 44
        self.assertEqual(self.one_digit[44], 0)
        self.assertEqual(self.vowel_one[44], 0)
        self.assertEqual(self.rwth[44], 0)
        self.assertEqual(self.underscores[44], 0)
        self.assertEqual(self.digits[44], 6 / 17)

    def test_scaling(self):
        # TODO
        pass