//
// Created by xabdomo on 4/13/25.
//

#include "cli.hpp"

#include <utility>


static std::vector<std::string> tokenize(const std::string &line) {
    std::vector<std::string> tokens;
    std::string token;
    bool inQuotes = false;
    bool escaping = false;
    
    for (char ch : line) {
        if (escaping) {
            // Append the character literally regardless of its type
            token.push_back(ch);
            escaping = false;
            continue;
        }
        
        if (ch == '\\') {
            escaping = true;
            continue;
        }

        if (ch == '"') {
            inQuotes = !inQuotes;
            continue; // Do not include the quote character
        }
        
        // If not in quotes and hit whitespace, then token boundary
        if (!inQuotes && std::isspace(static_cast<unsigned char>(ch))) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        }
        else {
            token.push_back(ch);
        }
    }
    
    // Append any remaining token
    if (!token.empty())
        tokens.push_back(token);
    
    return tokens;
}


CLI::CLI(CmdHandler fallbackFunction) : fallback(std::move(fallbackFunction)), running(false) {}

void CLI::addCommand(const std::string &command, CmdHandler handler) {
    // We store the command keyword (e.g. "load" or "copy")
    commands[command] = std::move(handler);
}

void CLI::run() {
    running = true;
    while (running) {
        std::cout << "> "; // prompt
        std::string line;
        if (!std::getline(std::cin, line))
            break;  // end-of-input

        line.erase(line.begin(), std::ranges::find_if(line,
                                                      [](const int ch) { return !std::isspace(ch); }));
        line.erase(std::find_if(line.rbegin(), line.rend(),
                                [](const int ch) { return !std::isspace(ch); }).base(),
                   line.end());
        if (line.empty()) continue;

        std::vector<std::string> tokens = tokenize(line);
        if (tokens.empty()) continue;
        const std::string cmd = tokens[0];

        if (cmd == "exit") {
            running = false;
            continue;
        }

        tokens.erase(tokens.begin());

        auto it = commands.find(cmd);
        if (it != commands.end()) {
            it->second(tokens);
        }
        else {
            // Command not found – call fallback.
            fallback({ line });
        }
    }
}

void CLI::exit() { running = false; }
