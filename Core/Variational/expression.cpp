#include "Core/Variational/expression.h"
#include <cmath>
#include <exception>
#include <algorithm>

using namespace QPanda::Variational;

expression::expression(var _root) : root(_root){}

var expression::getRoot() const{
    return root;
}

std::vector<var> expression::findLeaves(){
    std::vector<var> leaves;
    std::queue<var> q;
    q.push(root);

    while(!q.empty()){
        var v = q.front();
        if(v.getChildren().empty()){
            leaves.emplace_back(v);
        }
        else{
            std::vector<var> children = v.getChildren();
            for(const var& v : children)
                q.push(v);
        }
        q.pop();
    }
    std::vector<var> ans;
    std::copy(leaves.begin(), leaves.end(), std::back_inserter(ans));
    return ans;
}

std::vector<var> expression::findVariables() {
    std::vector<var> leaves;
    std::queue<var> q;
    q.push(root);

    while (!q.empty()) {
        var v = q.front();
        if (v.getChildren().empty() && v.getValueType()) {
            leaves.emplace_back(v);
        }
        else {
            std::vector<var> children = v.getChildren();
            for (const var& v : children)
                q.push(v);
        }
        q.pop();
    }
    return leaves;
}

void _rpropagate(var& v){
    if(v.getChildren().empty())
        return;
    std::vector<var> children = v.getChildren(); 
    for(var& _v : children){
        _rpropagate(_v);
    }
    v.setValue(v._eval());
}

MatrixXd expression::propagate(){
    _rpropagate(root);
    return root.getValue();
}

MatrixXd expression::propagate(const std::vector<var>& leaves){
    std::queue<var> q;
    std::unordered_map<var, int> explored; 
    for(const var& v : leaves){
        q.push(v); 
    }
    while(!q.empty()){
        var v = q.front();
        q.pop();
        std::vector<var> parents = v.getParents();
        for(var& parent : parents){
            explored[parent]++; 
            if(parent.getNumOpArgs() == explored[parent]){
                parent.setValue(parent._eval());
                q.push(parent);
            }
        } 
    } 
    return root.getValue();
}

std::vector<var> expression::findNonConsts(const std::vector<var>& leaves){
    std::vector<var> nonconsts;
    std::queue<var> q; 
    for(const var& v : leaves)
        q.push(v); 

    while(!q.empty()){
        var v = q.front();
        q.pop();

        if (std::end(nonconsts) != std::find(std::begin(nonconsts), std::end(nonconsts), v))
            continue;
        
        nonconsts.emplace_back(v);
        std::vector<var> parents = v.getParents();
        for(const var& parent : parents){
            q.push(parent);
        }
    }
    return nonconsts;
}

//copied from std::vector<var> expression::findNonConsts(const std::vector<var>& leaves)
std::vector<var> expression::findNonConsts(const std::unordered_set<var>& leaves) {
    std::vector<var> nonconsts;
    std::queue<var> q;
    for (const var& v : leaves)
        q.push(v);

    while (!q.empty()) {
        var v = q.front();
        q.pop();

        // We should not traverse this if it has already been visited.
        if (std::end(nonconsts) != std::find(std::begin(nonconsts), std::end(nonconsts), v))
            continue;

        nonconsts.emplace_back(v);
        std::vector<var> parents = v.getParents();
        for (const var& parent : parents) {
            q.push(parent);
        }
    }
    return nonconsts;
}

void expression::backpropagate(std::unordered_map<var, MatrixXd>& leaves){
    std::queue<var> q;
    std::unordered_map<var, MatrixXd> derivatives;
    std::unordered_map<var, size_t> explored;
    q.push(root);
    derivatives[root] = ones_like(root);
    
    while(!q.empty()){
        var v = q.front();
        q.pop();
        std::vector<var>& children = v.getChildren();
        std::vector<MatrixXd> child_derivs = v._back(derivatives[v]);
        for(size_t i = 0; i < children.size(); i++){
            auto child = children[i];
            if(explored.find(child) == explored.end())
                explored[child] = child.getParents().size();
            explored[child]--;
            if(derivatives.find(child) == derivatives.end())
                derivatives.emplace(child, zeros_like(child));
            derivatives[child] = derivatives[child].array() + child_derivs[i].array();
            if(children[i].getOp() != op_type::none && explored[child] == 0)
                q.push(child); 
        }
    }
   
    for(auto& iter : leaves){
        iter.second = derivatives[iter.first]; 
    } 
}

void expression::backpropagate(std::unordered_map<var, MatrixXd>& leaves, 
        const std::vector<var>& nonconsts){
    std::queue<var> q;
    std::unordered_map<var, MatrixXd> derivatives;
    std::unordered_map<var, size_t> explored;

    q.push(root);
    derivatives[root] = ones_like(root);
    
    while(!q.empty()){
        var v = q.front();
        q.pop();

        if (std::end(nonconsts) == std::find(std::begin(nonconsts), std::end(nonconsts), v))
            continue;

        std::vector<var>& children = v.getChildren();
        std::vector<MatrixXd> child_derivs = v._back(derivatives[v], nonconsts);
		for(size_t i = 0; i < children.size(); i++){
			auto child = children[i];
            if(explored.find(child) == explored.end())
                explored[child] = child.getParents().size();
            explored[child]--;
            // Be careful to not override the derivative value!
            if(derivatives.find(child) == derivatives.end())
                derivatives.emplace(child, zeros_like(child));
            derivatives[child] = derivatives[child].array() + child_derivs[i].array();
            if(children[i].getOp() != op_type::none && explored[child] == 0)
                q.push(child); 
        }
    }
   
    for(auto& iter : leaves){
        iter.second = derivatives[iter.first]; 
    } 
}

