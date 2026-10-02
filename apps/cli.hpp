#pragma once
// Command-line parsing shared by the apps. It is deliberately strict: a
// typo such as "--simulation 800" or "--games 4O" must stop the program with a
// message naming the option, not silently run with a default -- a mistake
// like that can waste an overnight training run. Header-only because only the
// apps (and test_cli) use it; it is not part of the azgo library.
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

// Strict "--name value" / "--flag" parser. Unknown options, missing values
// and malformed numbers throw std::invalid_argument with the option name.
class ArgParser {
public:
    // valueOptions take a value; flagOptions do not. "--set" may repeat.
    ArgParser(int argc, char** argv, const std::vector<std::string>& valueOptions,
              const std::vector<std::string>& flagOptions);
    bool has(const std::string& name) const;
    std::string get(const std::string& name, const std::string& def = "") const;
    int getInt(const std::string& name, int def) const;       // strtol, whole string
    double getDouble(const std::string& name, double def) const;
    bool flag(const std::string& name) const;
    const std::vector<std::string>& sets() const { return sets_; }   // every --set value
private:
    std::map<std::string, std::string> values_;
    std::vector<std::string> flags_;
    std::vector<std::string> sets_;
};

inline ArgParser::ArgParser(int argc, char** argv, const std::vector<std::string>& valueOptions,
                            const std::vector<std::string>& flagOptions) {
    auto contains = [](const std::vector<std::string>& v, const std::string& s) {
        return std::find(v.begin(), v.end(), s) != v.end();
    };
    // A value may begin with '-' only when it is a number ("--komi -6.5");
    // otherwise "--run --resume" would swallow the next option as a value.
    auto isNumber = [](const std::string& s) {
        if (s.empty()) return false;
        char* end = nullptr;
        std::strtod(s.c_str(), &end);
        return end != s.c_str() && *end == '\0';
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.size() < 3 || arg.compare(0, 2, "--") != 0)
            throw std::invalid_argument("unexpected argument '" + arg + "'");
        const std::string name = arg.substr(2);
        if (contains(flagOptions, name)) {
            if (!contains(flags_, name)) flags_.push_back(name);
            continue;
        }
        if (!contains(valueOptions, name))
            throw std::invalid_argument("unknown option " + arg);
        if (i + 1 >= argc)
            throw std::invalid_argument("option " + arg + " requires a value");
        const std::string value = argv[++i];
        if (!value.empty() && value[0] == '-' && !isNumber(value))
            throw std::invalid_argument("option " + arg + " requires a value");
        if (name == "set") {
            sets_.push_back(value);
        } else {
            if (values_.count(name))
                throw std::invalid_argument("option " + arg + " given more than once");
            values_[name] = value;
        }
    }
}

inline bool ArgParser::has(const std::string& name) const {
    return values_.count(name) > 0 || (name == "set" && !sets_.empty());
}

inline std::string ArgParser::get(const std::string& name, const std::string& def) const {
    auto it = values_.find(name);
    return it == values_.end() ? def : it->second;
}

inline int ArgParser::getInt(const std::string& name, int def) const {
    auto it = values_.find(name);
    if (it == values_.end()) return def;
    const std::string& s = it->second;
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE || v < INT_MIN || v > INT_MAX)
        throw std::invalid_argument("option --" + name + " expects an integer, got '" + s + "'");
    return static_cast<int>(v);
}

inline double ArgParser::getDouble(const std::string& name, double def) const {
    auto it = values_.find(name);
    if (it == values_.end()) return def;
    const std::string& s = it->second;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || end != s.c_str() + s.size() || errno == ERANGE || !std::isfinite(v))
        throw std::invalid_argument("option --" + name + " expects a number, got '" + s + "'");
    return v;
}

inline bool ArgParser::flag(const std::string& name) const {
    return std::find(flags_.begin(), flags_.end(), name) != flags_.end();
}
