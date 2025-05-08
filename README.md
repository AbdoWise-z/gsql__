# gsql++

**gsql++** is a modern C++ dbms designed to provide a lightweight SQL-like query interface for in-memory data structures. Built with modularity and performance in mind using GPU for acceleration, it uses CMake for configuration and build management.

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

Build the full project

```bash
./build.sh
```


### Running

Run the compiled binary:

```bash
./gsql__ [inputs folder] [query file]
```

Example:
```bash
./gsql__ ./data_test ./data_test/q1.txt
```


## Project Structure

```
gsql++/
├── data_test/     # Sample data to test with
├── src/           # Source files
├── libs/          # project libraries
├── deps/          # project dependancies
├── CMakeLists.txt # CMake build configuration
├── build.sh
├── build-project.sh
├── build-deps.sh
└── README.md
```

## License

MIT License
