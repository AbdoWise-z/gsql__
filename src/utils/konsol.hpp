//
// Created by xabdomo on 4/15/25.
//

#ifndef KONSOL_HPP
#define KONSOL_HPP

#include <string>

// ANSI escape codes for colors and cursor control.
#define SELECT_BG "\033[48;5;240m"  // Dark gray background for selected variable.
#define HEADER_BG "\033[41m"        // Red background for stack headers.
#define RESET_COLOR "\033[0m"

#define BLACK_BG   "\033[40m"    // Black background
#define RED_BG     "\033[41m"    // Red background
#define GREEN_BG   "\033[42m"    // Green background
#define YELLOW_BG  "\033[43m"    // Yellow background
#define BLUE_BG    "\033[44m"    // Blue background
#define MAGENTA_BG "\033[45m"    // Magenta background
#define CYAN_BG    "\033[46m"    // Cyan background
#define WHITE_BG   "\033[47m"    // White background
#define RESET_BG   "\033[49m"    // Reset to default background color

#define BLACK_FG   "\033[30m"    // Black text
#define RED_FG     "\033[31m"    // Red text
#define GREEN_FG   "\033[32m"    // Green text
#define YELLOW_FG  "\033[33m"    // Yellow text
#define BLUE_FG    "\033[34m"    // Blue text
#define MAGENTA_FG "\033[35m"    // Magenta text
#define CYAN_FG    "\033[36m"    // Cyan text
#define WHITE_FG   "\033[37m"    // White text
#define RESET_FG   "\033[39m"    // Reset to default text color

#define BRIGHT_BLACK_FG   "\033[90m"  // Bright Black text
#define BRIGHT_RED_FG     "\033[91m"  // Bright Red text
#define BRIGHT_GREEN_FG   "\033[92m"  // Bright Green text
#define BRIGHT_YELLOW_FG  "\033[93m"  // Bright Yellow text
#define BRIGHT_BLUE_FG    "\033[94m"  // Bright Blue text
#define BRIGHT_MAGENTA_FG "\033[95m"  // Bright Magenta text
#define BRIGHT_CYAN_FG    "\033[96m"  // Bright Cyan text
#define BRIGHT_WHITE_FG   "\033[97m"  // Bright White text

#define BRIGHT_BLACK_BG   "\033[100m" // Bright Black background
#define BRIGHT_RED_BG     "\033[101m" // Bright Red background
#define BRIGHT_GREEN_BG   "\033[102m" // Bright Green background
#define BRIGHT_YELLOW_BG  "\033[103m" // Bright Yellow background
#define BRIGHT_BLUE_BG    "\033[104m" // Bright Blue background
#define BRIGHT_MAGENTA_BG "\033[105m" // Bright Magenta background
#define BRIGHT_CYAN_BG    "\033[106m" // Bright Cyan background
#define BRIGHT_WHITE_BG   "\033[107m" // Bright White background

#define BOLD       "\033[1m"    // Bold text
#define UNDERLINE  "\033[4m"    // Underlined text
#define REVERSED   "\033[7m"    // Inverted text
#define RESET_ATTR "\033[0m"    // Reset all attributes (bold, underlined, etc.)

// ANSI codes to hide/show cursor.
#define HIDE_CURSOR "\033[?25l"
#define SHOW_CURSOR "\033[?25h"
#include <string>

inline std::string fcolor(const std::string &text, const std::string &color) {
    return color + text + RESET_FG;
}

inline std::string bcolor(const std::string &text, const std::string &color) {
    return color + text + RESET_FG;
}

inline std::string color(const std::string &text, const std::string &color) {
    return color + text + RESET_COLOR;
}

inline std::string theme(const std::string &text, const std::string &theme) {
    return theme  + text + RESET_ATTR;
}

inline std::string bold(const std::string& text) {
    return BOLD + text + RESET_ATTR;
}

inline std::string underline(const std::string& text) {
    return UNDERLINE + text + RESET_ATTR;
}


#endif //KONSOL_HPP
