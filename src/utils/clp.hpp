//
// Created by xabdomo on 4/16/25.
//

#ifndef CLP_HPP
#define CLP_HPP

#include <vector>
#include <string>
#include <map>
#include <variant>
#include <stdexcept>
#include <algorithm>
#include <cctype>

class CommandLineParser {
public:
    enum class Type { INT, STRING, BOOL, FLOAT };

    void addArgument(const std::string& name, Type type, bool isOptional = false, const std::variant<int, std::string, bool, float>& defaultValue = {}) {
        if (isOptional) {
            bool valid = false;
            switch (type) {
                case Type::INT: valid = std::holds_alternative<int>(defaultValue); break;
                case Type::STRING: valid = std::holds_alternative<std::string>(defaultValue); break;
                case Type::BOOL: valid = std::holds_alternative<bool>(defaultValue); break;
                case Type::FLOAT: valid = std::holds_alternative<float>(defaultValue); break;
            }
            if (!valid) {
                throw std::invalid_argument("Default value type mismatch for argument: " + name);
            }
        }
        allowedArgs[name] = {type, isOptional, defaultValue};
    }

    void parse(int argc, char* argv[]) {
        parsedArgs.clear();

        // Set defaults for optional arguments
        for (const auto& [name, arg] : allowedArgs) {
            if (arg.isOptional) {
                parsedArgs[name] = arg.defaultValue;
            }
        }

        int i = 1;
        while (i < argc) {
            std::string token = argv[i];
            if (token.size() >= 2 && token.substr(0, 2) == "--") {
                std::string argName = token.substr(2);
                auto it = allowedArgs.find(argName);
                if (it == allowedArgs.end()) {
                    throw std::runtime_error("Unrecognized argument: --" + argName);
                }

                const auto& arg = it->second;
                i++; // Move past argument name

                if (arg.type == Type::BOOL) {
                    bool value = true;
                    if (i < argc) {
                        std::string nextToken = argv[i];
                        std::transform(nextToken.begin(), nextToken.end(), nextToken.begin(),
                            [](unsigned char c){ return std::tolower(c); });
                        if (nextToken == "true") {
                            value = true;
                            i++;
                        } else if (nextToken == "false") {
                            value = false;
                            i++;
                        }
                    }
                    parsedArgs[argName] = value;
                } else {
                    if (i >= argc) {
                        throw std::runtime_error("Missing value for argument: --" + argName);
                    }
                    std::string valueStr = argv[i];
                    i++;

                    try {
                        switch (arg.type) {
                            case Type::INT:
                                parsedArgs[argName] = std::stoi(valueStr);
                                break;
                            case Type::FLOAT:
                                parsedArgs[argName] = std::stof(valueStr);
                                break;
                            case Type::STRING:
                                parsedArgs[argName] = valueStr;
                                break;
                            default:
                                throw std::runtime_error("Unexpected type");
                        }
                    } catch (const std::exception& e) {
                        throw std::runtime_error("Invalid value for argument --" + argName + ": " + valueStr);
                    }
                }
            } else {
                throw std::runtime_error("Invalid argument format: " + token);
            }
        }

        // Validate required arguments
        for (const auto& [name, arg] : allowedArgs) {
            if (!arg.isOptional && parsedArgs.find(name) == parsedArgs.end()) {
                throw std::runtime_error("Missing required argument: --" + name);
            }
        }
    }

    template<typename T>
    T get(const std::string& name) const {
        auto it = parsedArgs.find(name);
        if (it == parsedArgs.end()) {
            throw std::runtime_error("Argument not found: " + name);
        }
        return std::get<T>(it->second);
    }

private:
    struct Argument {
        Type type;
        bool isOptional;
        std::variant<int, std::string, bool, float> defaultValue;
    };

    std::map<std::string, Argument> allowedArgs;
    std::map<std::string, std::variant<int, std::string, bool, float>> parsedArgs;
};


#endif //CLP_HPP
