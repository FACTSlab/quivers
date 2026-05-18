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

    /* EOF token: emit once at end of input so the outer grammar can
     * close any still-open blocks. */
    if (lexer->eof(lexer) && valid_symbols[EOF_TOKEN]) {
        lexer->result_symbol = EOF_TOKEN;
        return true;
    }

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
