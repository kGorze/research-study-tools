#include <vector>
#include <string>
#include <algorithm>
#include <map>
using namespace std;

class Solution {
public:
    vector<int> topKFrequent1(vector<int>& nums, int k) {
       map<int, int> freq; 
        for(auto& num : nums) {
            freq[num]++;
        }
        vector<pair<int, int>> freq_vec;
        
        for(auto& [key, value] : freq) {
            freq_vec.push_back({value, key});
        }
        sort(freq_vec.begin(), freq_vec.end(), greater<pair<int, int>>());
        vector<int> result;
        for(int i = 0; i < k; i++) {
            result.push_back(freq_vec[i].second);
        }
        return result;
        // Time complexity: O(nlogn)
        // Space complexity: O(n)
    }
    vector<int> topKFrequent(vector<int>& nums, int k) {
    
    }
};
