
# Filter PCREs to icgrep, pcre2mnrl and vasim

# export LOG=DEBUG
#### ANMLZoo

# Brill
echo "###########################"
echo "Brill"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Brill/regex/brill.1chip.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Brill/regex/brill.1chip.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Brill/regex/brill.1chip.icgrep.anml.regex

# CAV
echo "###########################"
echo "CAV"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/ClamAV/regex/515_nocounter.1chip.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/ClamAV/regex/515_nocounter.1chip.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/ClamAV/regex/515_nocounter.1chip.icgrep.anml.regex

# Dotstar
echo "###########################"
echo "Dotstar"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Dotstar/regex/backdoor_dotstar.1chip.regex.pcre
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Dotstar/regex/backdoor_dotstar.1chip.regex.icgrep.pcre
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Dotstar/regex/backdoor_dotstar.1chip.regex.icgrep.anml.pcre

# Protomata
echo "###########################"
echo "Protomata"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Protomata/regex/2340sigs.1chip.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Protomata/regex/2340sigs.1chip.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Protomata/regex/2340sigs.1chip.icgrep.anml.regex

# Snort
echo "###########################"
echo "Snort"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Snort/regex/snort.1chip.fix.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Snort/regex/snort.1chip.fix.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/ANMLZoo/Snort/regex/snort.1chip.fix.icgrep.anml.regex


#### AutomataZoo

#YARA
echo "###########################"
echo "YARA"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/AutomataZoo/YARA/code/YARA_wide_rules.txt
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/AutomataZoo/YARA/code/YARA_wide_rules.icgrep.txt
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/AutomataZoo/YARA/code/YARA_wide_rules.icgrep.anml.txt

# python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/AutomataZoo/YARA/code/YARA_rules.txt
# python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/AutomataZoo/YARA/code/YARA_rules.icgrep.txt
# python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/AutomataZoo/YARA/code/YARA_rules.icgrep.anml.txt

#### Regex

# Bro217
echo "###########################"
echo "Bro217"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/Regex/Bro217/regex/bro217.re.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/Regex/Bro217/regex/bro217.re.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/Regex/Bro217/regex/bro217.re.icgrep.anml.regex


# ExactMath
echo "###########################"
echo "ExactMath"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/Regex/ExactMath/regex/exact-math.conf_300-0.re.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/Regex/ExactMath/regex/exact-math.conf_300-0.re.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/Regex/ExactMath/regex/exact-math.conf_300-0.re.icgrep.anml.regex

# Ranges1
echo "###########################"
echo "Ranges1"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/Regex/Ranges1/regex/ranges1.conf_300-0.re.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/Regex/Ranges1/regex/ranges1.conf_300-0.re.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/Regex/Ranges1/regex/ranges1.conf_300-0.re.icgrep.anml.regex

# TCP
echo "###########################"
echo "TCP"
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_icgrep.py -f=${BITGEN_ROOT}/datasets/Regex/TCP/regex/ranges05.conf_300-0.re.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_pcre_anml.py -f=${BITGEN_ROOT}/datasets/Regex/TCP/regex/ranges05.conf_300-0.re.icgrep.regex
python ${BITGEN_ROOT}/scripts/datasets/filter_unique.py -f=${BITGEN_ROOT}/datasets/Regex/TCP/regex/ranges05.conf_300-0.re.icgrep.anml.regex