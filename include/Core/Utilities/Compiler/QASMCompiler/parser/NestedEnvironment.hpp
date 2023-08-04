#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

template <typename T> class NestedEnvironment {
private:
  std::vector<std::map<std::string, T>> env{};

public:
  NestedEnvironment() { env.push_back({}); };

  void push() { env.push_back({}); }

  void pop() { env.pop_back(); }

  std::optional<T> find(std::string key) {
    for (auto it = env.rbegin(); it != env.rend(); ++it) {
      auto found = it->find(key);
      if (found != it->end()) {
        return found->second;
      }
    }
    return std::nullopt;
  }

  void emplace(std::string key, T value) { env.back().emplace(key, value); }
};
