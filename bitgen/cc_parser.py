
# Generated from VASim by ChatGPT

class Parser:
    OPEN_BRACKET = 256  # Represents an open bracket outside the 0-255 range

    def __init__(self):
        # Initialize a column with eight 32-bit unsigned integers set to 0
        self.column = [0] * 8  # Each element represents 32 bits

    def set_bit(self, bit, value=1):
        """
        Sets or clears a specific bit in the column.

        :param bit: Integer from 0 to 255 representing the bit position.
        :param value: 1 to set the bit, 0 to clear the bit.
        """
        if not (0 <= bit <= 255):
            raise ValueError(f"Bit index out of range: {bit}")

        uint_index = bit // 32
        bit_index = bit % 32
        if value:
            self.column[uint_index] |= (1 << bit_index)
        else:
            self.column[uint_index] &= ~(1 << bit_index)

    def get_bit(self, bit):
        """
        Retrieves the value of a specific bit in the column.

        :param bit: Integer from 0 to 255 representing the bit position.
        :return: 1 if the bit is set, 0 otherwise.
        """
        if not (0 <= bit <= 255):
            raise ValueError(f"Bit index out of range: {bit}")

        uint_index = bit // 32
        bit_index = bit % 32
        return (self.column[uint_index] >> bit_index) & 1

    def set_range(self, start, end, value=1):
        """
        Sets or clears a range of bits from start to end inclusive.

        :param start: Starting bit index (0-255).
        :param end: Ending bit index (0-255).
        :param value: 1 to set the bits, 0 to clear the bits.
        """
        if not (0 <= start <= end <= 255):
            raise ValueError(f"Invalid range: start={start}, end={end}")

        for bit in range(start, end + 1):
            self.set_bit(bit, value)

    def flip_all_bits(self):
        """
        Inverts all bits in the column.
        """
        for i in range(8):
            self.column[i] ^= 0xFFFFFFFF

    def parse_symbol_set(self, symbol_set):
        """
        Parses a symbol set string and sets corresponding bits in the column.

        :param symbol_set: String representing the symbol/character set to parse.
        """
        # Reset the column
        self.column = [0] * 8

        if symbol_set == "*":
            # Set all bits to 1
            for i in range(8):
                self.column[i] = 0xFFFFFFFF
            return

        if symbol_set == ".":
            # Set all bits to 1 except for '\n' (ASCII 10)
            for i in range(8):
                self.column[i] = 0xFFFFFFFF
            self.set_bit(ord('\n'), 0)
            return

        # Handle symbol sets that start and end with curly braces {###}
        if symbol_set.startswith('{') and symbol_set.endswith('}'):
            print("CURLY BRACES NOT IMPLEMENTED")
            raise NotImplementedError("Curly braces are not implemented.")

        escaped = False
        inverting = False
        range_set = False
        bracket_sem = 0
        brace_sem = 0
        last_char = 0
        range_start = 0

        index = 0
        while index < len(symbol_set):
            c = symbol_set[index]
            c_ord = ord(c)

            if c_ord > 255:
                raise ValueError(f"Character code out of range: {c}")

            if c == '[':
                if escaped:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    last_char = c_ord
                    escaped = False
                else:
                    last_char = self.OPEN_BRACKET
                    bracket_sem += 1

            elif c == ']':
                if escaped:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    escaped = False
                    last_char = c_ord
                else:
                    bracket_sem -= 1

            elif c == '{':
                # Curly braces are treated as literal characters
                self.set_bit(c_ord, 1)
                if range_set:
                    self.set_range(range_start, c_ord, 1)
                    range_set = False
                last_char = c_ord

            elif c == '}':
                # Curly braces are treated as literal characters
                self.set_bit(c_ord, 1)
                if range_set:
                    self.set_range(range_start, c_ord, 1)
                    range_set = False
                last_char = c_ord

            elif c == '\\':
                if escaped:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    last_char = c_ord
                    escaped = False
                else:
                    escaped = True

            elif c in {'n', 'r', 't', 'a', 'b', 'f', 'v', '\'', '\"'}:
                if escaped:
                    escape_dict = {
                        'n': ord('\n'),
                        'r': ord('\r'),
                        't': ord('\t'),
                        'a': ord('\a'),
                        'b': ord('\b'),
                        'f': ord('\f'),
                        'v': ord('\v'),
                        '\'': ord('\''),
                        '\"': ord('\"')
                    }
                    actual_char = escape_dict.get(c, c_ord)
                    self.set_bit(actual_char, 1)
                    if range_set:
                        self.set_range(range_start, actual_char, 1)
                        range_set = False
                    last_char = actual_char
                    escaped = False
                else:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    last_char = c_ord

            elif c == '-':
                if escaped or last_char == self.OPEN_BRACKET:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    escaped = False
                    last_char = c_ord
                else:
                    range_set = True
                    range_start = last_char

            elif c in {'s', 'd', 'w'}:
                if escaped:
                    if c == 's':
                        # Set whitespace characters: '\n', '\t', '\r', '\v', '\f', ' '
                        for whitespace_char in [ord('\n'), ord('\t'), ord('\r'), 11, 12, 32]:
                            self.set_bit(whitespace_char, 1)
                    elif c == 'd':
                        # Set digits '0'-'9'
                        self.set_range(48, 57, 1)
                    elif c == 'w':
                        # Set word characters '_', '0'-'9', 'A'-'Z', 'a'-'z'
                        self.set_bit(ord('_'), 1)
                        self.set_range(48, 57, 1)   # '0'-'9'
                        self.set_range(65, 90, 1)   # 'A'-'Z'
                        self.set_range(97, 122, 1)  # 'a'-'z'
                    range_set = False
                    last_char = 0  # Reset last_char after setting a class
                    escaped = False
                else:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    last_char = c_ord

            elif c == '^':
                if escaped:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    last_char = c_ord
                    escaped = False
                else:
                    inverting = True

            elif c == 'x':
                if escaped:
                    index += 1
                    if index + 1 > len(symbol_set):
                        raise ValueError("Incomplete hex escape sequence")
                    hex_str = symbol_set[index:index+2]
                    try:
                        number = int(hex_str, 16)
                    except ValueError:
                        raise ValueError(f"Invalid hex escape sequence: \\x{hex_str}")
                    self.set_bit(number, 1)
                    if range_set:
                        self.set_range(range_start, number, 1)
                        range_set = False
                    last_char = number
                    escaped = False
                    index += 1  # Skip the next hex digit
                else:
                    self.set_bit(c_ord, 1)
                    if range_set:
                        self.set_range(range_start, c_ord, 1)
                        range_set = False
                    last_char = c_ord

            else:
                if escaped:
                    # Treat unknown escape sequences as literal characters
                    escaped = False
                self.set_bit(c_ord, 1)
                if range_set:
                    self.set_range(range_start, c_ord, 1)
                    range_set = False
                last_char = c_ord

            index += 1

        if inverting:
            self.flip_all_bits()

        if bracket_sem != 0 or brace_sem != 0:
            print(f"MALFORMED BRACKETS OR BRACES: {symbol_set}")
            print(f"brackets: {bracket_sem}")
            raise ValueError("Malformed brackets or braces in symbol set.")

    def get_set_characters(self):
        """
        Retrieves a list of characters whose corresponding bits are set.

        :return: List of characters with bits set to 1.
        """
        set_chars = []
        for bit in range(256):
            if self.get_bit(bit):
                set_chars.append(chr(bit))
        return set_chars

    def display_column(self):
        """
        Displays the column as eight 32-bit unsigned integers in hexadecimal.
        """
        print("Column as uint32[8]:")
        for i, uint in enumerate(self.column):
            print(f"uint32[{i}]: 0x{uint:08X}")

# Example Usage
if __name__ == "__main__":
    parser = Parser()

    try:
        # Define various symbol sets to test
        test_symbol_sets = [
            "*",            # Set all bits
            ".",            # Set all bits except '\n'
            "\\d",          # Set digits '0'-'9'
            "\\w",          # Set word characters
            "\\s",          # Set whitespace characters
            "A-Z",          # Set uppercase letters
            "a-z",          # Set lowercase letters
            "0-9A-Fa-f",    # Set hexadecimal digits
            "\\x41\\x42",   # Set 'A' and 'B' using hex
            "^\\d",         # Invert digits
            "[A-Z]",        # Set uppercase letters within brackets
            "[\\w-]"        # Set word characters and hyphen within brackets
            # Add more test cases as needed
        ]

        for symbol_set in test_symbol_sets:
            print(f"\nParsing symbol set: {symbol_set}")
            parser.parse_symbol_set(symbol_set)
            parser.display_column()
            set_chars = parser.get_set_characters()
            print(f"Set characters ({len(set_chars)}): {set_chars}")

    except NotImplementedError as nie:
        print(nie)
    except ValueError as ve:
        print(ve)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
