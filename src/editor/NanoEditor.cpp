//
// Created by xabdomo on 3/27/25.
//

#include "NanoEditor.h"

#include <unistd.h>
#include <iostream>
#include <fstream>

std::string NanoEditor::edit() {
    std::string tempFileName = "/tmp/nano_temp_" + std::to_string(getpid()) + ".cmm";

    std::string command = "xterm -fa \"DejaVu Sans Mono\" -fs 16 -e nano -i -Y java -T 4 " + tempFileName;
    int result = system(command.c_str());

    std::ifstream file(tempFileName);
    std::string content;
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            content += line + "\n";
        }
        file.close();

        // Remove the temporary file
        std::remove(tempFileName.c_str());

        return content;
    }

    return "";
}
