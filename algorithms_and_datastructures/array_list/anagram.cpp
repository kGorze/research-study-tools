#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>

class Solution {
public:
    std::vector<std::vector<std::string>> groupAnagrams(std::vector<std::string>& strs) {
        std::unordered_map<std::string, std::vector<std::string>> anagrams;
        for(auto& str : strs) {
            std::string sorted_str = str;
            std::sort(sorted_str.begin(), sorted_str.end());
            anagrams[sorted_str].push_back(str);
        }
        std::vector<std::vector<std::string>> result;
        for(auto& [key, value] : anagrams) {
            result.push_back(value);
        }
        return result;
    }
};
