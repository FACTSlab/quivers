/* External scanner for QVR's Python-style indentation layout.
 *
 * Emits four tokens:
 *
 *   NEWLINE  end of a statement at the current indent level
 *   INDENT   the indent column went up; opens a block
 *   DEDENT   the indent column went down; closes a block
 *   EOF      sentinel emitted once at end of input
 *
 * Adapted from the tree-sitter-python reference scanner; string
 * handling stripped out (QVR uses a regex-based string token directly
 * in grammar.js), comment handling stripped (QVR's comments are
 * tree-sitter `extras` and never carry an indent contribution).
 */

#include "tree_sitter/array.h"
#include "tree_sitter/parser.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum TokenType {
    NEWLINE,
    INDENT,
    DEDENT,
    EOF_TOKEN,
};

typedef struct {
    Array(uint16_t) indents;
} Scanner;

static inline void advance(TSLexer *lexer) { lexer->advance(lexer, false); }
static inline void skip(TSLexer *lexer) { lexer->advance(lexer, true); }

bool tree_sitter_qvr_external_scanner_scan(
    void *payload, TSLexer *lexer, const bool *valid_symbols
) {
    Scanner *scanner = (Scanner *)payload;

    /* At true end of input we must make progress on every call or
     * tree-sitter re-enters the scanner forever. Drain still-open
     * blocks with DEDENTs first, then commit to EOF_TOKEN when it is
     * a valid alternative (the outer ``source_file`` rule consumes
     * it). Only if the parser explicitly does not accept EOF here
     * do we fall back to a final NEWLINE; EOF_TOKEN being preferred
     * over NEWLINE at end-of-input is what breaks the zero-width
     * NEWLINE re-emission loop. */
    if (lexer->eof(lexer)) {
        if (valid_symbols[DEDENT] && scanner->indents.size > 1) {
            array_pop(&scanner->indents);
            lexer->result_symbol = DEDENT;
            return true;
        }
        if (valid_symbols[EOF_TOKEN]) {
            lexer->result_symbol = EOF_TOKEN;
            return true;
        }
        if (valid_symbols[NEWLINE]) {
            lexer->result_symbol = NEWLINE;
            return true;
        }
        return false;
    }

    /* Mark the start of the token: if we end up emitting a NEWLINE /
     * INDENT / DEDENT we will move ``mark_end`` forward past the
     * whitespace we consume so tree-sitter sees the token as
     * spanning those bytes. */
    lexer->mark_end(lexer);

    bool found_end_of_line = false;
    uint16_t indent_length = 0;

    /* Scan leading whitespace + newlines + comments to find the next
     * non-blank line's indentation level. */
    for (;;) {
        if (lexer->lookahead == '\n') {
            found_end_of_line = true;
            indent_length = 0;
            skip(lexer);
        } else if (lexer->lookahead == ' ') {
            indent_length++;
            skip(lexer);
        } else if (lexer->lookahead == '\r' || lexer->lookahead == '\f') {
            indent_length = 0;
            skip(lexer);
        } else if (lexer->lookahead == '\t') {
            /* Treat tabs as 8 spaces of indentation, matching Python. */
            indent_length += 8;
            skip(lexer);
        } else if (lexer->lookahead == '#') {
            /* Skip over comment lines without contributing to indent.
             * If we haven't yet seen a newline, this is a trailing
             * comment on a statement; bail out so the parser sees a
             * NEWLINE from the comment's own line terminator. */
            if (!found_end_of_line) {
                return false;
            }
            while (lexer->lookahead && lexer->lookahead != '\n') {
                skip(lexer);
            }
            /* The trailing newline of the comment is consumed by the
             * outer loop on the next iteration. */
            indent_length = 0;
        } else if (lexer->eof(lexer)) {
            indent_length = 0;
            found_end_of_line = true;
            break;
        } else {
            break;
        }
    }

    if (found_end_of_line) {
        /* Tree-sitter only consumes the bytes between the original
         * position and ``mark_end``. Advance ``mark_end`` past the
         * whitespace we just skipped so the emitted token actually
         * spans the newline; otherwise tree-sitter treats it as a
         * zero-width token and re-enters the scanner at the same
         * position, producing an infinite loop. */
        lexer->mark_end(lexer);

        if (scanner->indents.size > 0) {
            uint16_t current_indent_length = *array_back(&scanner->indents);

            if (valid_symbols[INDENT] && indent_length > current_indent_length) {
                array_push(&scanner->indents, indent_length);
                lexer->result_symbol = INDENT;
                return true;
            }

            if (valid_symbols[DEDENT] && indent_length < current_indent_length) {
                array_pop(&scanner->indents);
                lexer->result_symbol = DEDENT;
                return true;
            }
        }

        if (valid_symbols[NEWLINE]) {
            lexer->result_symbol = NEWLINE;
            return true;
        }
    }

    return false;
}

unsigned tree_sitter_qvr_external_scanner_serialize(void *payload, char *buffer) {
    Scanner *scanner = (Scanner *)payload;

    size_t size = 0;

    /* Serialize the indent stack, two bytes per entry (uint16_t LE).
     * Skip the implicit zero at the bottom of the stack; it is
     * reconstructed on deserialize. */
    uint32_t iter = 1;
    for (; iter < scanner->indents.size && size + 1 < TREE_SITTER_SERIALIZATION_BUFFER_SIZE; ++iter) {
        uint16_t indent_value = *array_get(&scanner->indents, iter);
        buffer[size++] = (char)(indent_value & 0xFF);
        buffer[size++] = (char)((indent_value >> 8) & 0xFF);
    }

    return size;
}

void tree_sitter_qvr_external_scanner_deserialize(
    void *payload, const char *buffer, unsigned length
) {
    Scanner *scanner = (Scanner *)payload;

    array_delete(&scanner->indents);
    array_push(&scanner->indents, 0);

    size_t size = 0;
    while (size + 1 < length) {
        uint16_t indent_value = (unsigned char)buffer[size]
            | ((unsigned char)buffer[size + 1] << 8);
        array_push(&scanner->indents, indent_value);
        size += 2;
    }
}

void *tree_sitter_qvr_external_scanner_create(void) {
    Scanner *scanner = calloc(1, sizeof(Scanner));
    array_init(&scanner->indents);
    tree_sitter_qvr_external_scanner_deserialize(scanner, NULL, 0);
    return scanner;
}

void tree_sitter_qvr_external_scanner_destroy(void *payload) {
    Scanner *scanner = (Scanner *)payload;
    array_delete(&scanner->indents);
    free(scanner);
}
