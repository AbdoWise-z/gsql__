# gsql++

**gsql++** is a modern C++ dmbs designed to provide a lightweight SQL-like query interface for in-memory data structures. Built with modularity and performance in mind using GPU for acceleration, it uses CMake for configuration and build management.

## Features

- Written in modern C++ (C++17/20)
- Modular and extensible architecture
- Cross-platform build using CMake
- Lightweight and dependency-free core
- GPU Acceleration

## Getting Started

### Prerequisites

- C++ compiler with C++17 or newer support (e.g., GCC 9+, Clang 10+, MSVC 2019+)
- NVCC
- CMake 3.28 or higher

### Building the Project

Clone the repository:

```bash
git clone https://github.com/AbdoWise-z/gsql__.git
cd gsql__
```

Build dependencies

```bash
mkdir build
cd build
cmake ..
make
```

To build with a specific compiler or set build type:

```bash
cmake -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release ..
```

### Running

Run the compiled binary:

```bash
./gsqlpp [arguments]
```

## Usage

Example code snippet:

```cpp
#include "gsqlpp/gsqlpp.h"

int main() {
    gsqlpp::Database db;
    db.execute("SELECT * FROM nodes WHERE value > 5;");
    return 0;
}
```

## Project Structure

```
gsqlpp/
├── include/       # Public headers
├── src/           # Source files
├── tests/         # Unit and integration tests
├── CMakeLists.txt # CMake build configuration
└── README.md
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

### Build and Test

To run tests:

```bash
make test
```

Or with CTest:

```bash
ctest
```

## License

MIT License

## Acknowledgments

- Inspired by SQL query syntax
- Built with simplicity and performance in mind
