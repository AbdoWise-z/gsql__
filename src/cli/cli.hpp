//
// Created by xabdomo on 4/13/25.
//

#ifndef CLI_HPP
#define CLI_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <algorithm>

class CLI {
public:
    // The command handler function type.
    using CmdHandler = std::function<void(const std::vector<std::string>&)>;

    // Constructor receives a fallback handler that is called if no command matches.
    CLI(CmdHandler fallbackFunction);

    // Add a command mapping
    void addCommand(const std::string &command, CmdHandler handler);

    // Start the CLI loop. It continuously reads input until exit is called.
    void run();

    // Alternative to calling "exit" from inside a command handler.
    void exit();

private:
    std::unordered_map<std::string, CmdHandler> commands;
    CmdHandler fallback;
    bool running;
};

#endif //CLI_HPP
