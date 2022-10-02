/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_GRAPH_TEMPLATE_LOGICSUM_BASE_H_
#define TENSORFLOW_CORE_GRAPH_TEMPLATE_LOGICSUM_BASE_H_

#include "tensorflow/core/graph/template_base.h"

namespace tensorflow {

class TemplateLogicSumBase: public TemplateBase { // 自定义的模板。用来圈定原图中，你想要替换的一部分子图
 public:
  // 在构造函数中，就指定了temp_nodes_： 该模板要匹配的各nodes类型。和每个op和其他op间的图关系
  TemplateLogicSumBase() {
    const TempNode n0 = {          // 代表子图中一个op节点
      .key = "greater_0",          // key随便定，是该模板内该节点的唯一标识
      .op = "Greater",
      .inputs = {"0","1"},                           // 输入op. 如果对接的是外部输入，n代表外部第n个输入
      .outputs = {{"logic_or_0","logic_and_0"}}      // 对接的下一个op.  如果是内部op,写对应的key. 否则代表第n个外部输出
    };
    temp_nodes_.emplace_back(n0);

    const TempNode n1 = {
      .key = "greater_1",
      .op = "Greater",
      .inputs = {"2","3"},
      .outputs = {{"logic_or_0","logic_and_0"}}
    };
    temp_nodes_.emplace_back(n1);

    const TempNode n2 = {
      .key = "logic_or_0",
      .op = "LogicalOr",
      .inputs = {"greater_0","greater_1"},
      .outputs = {{"logic_xor_0"}}
    };
    temp_nodes_.emplace_back(n2);

    const TempNode n3 = {
      .key = "logic_and_0",
      .op = "LogicalAnd",
      .inputs = {"greater_0","greater_1"},
      .outputs = {{"logic_not_0"}}
    };
    temp_nodes_.emplace_back(n3);

    const TempNode n4 = {
      .key = "logic_not_0",
      .op = "LogicalNot",
      .inputs = {"logic_and_0"},
      .outputs = {{"logic_xor_0"}}
    };
    temp_nodes_.emplace_back(n4);

    const TempNode n5 = {
      .key = "logic_xor_0",
      .op = "LogicalAnd",
      .inputs = {"logic_or_0","logic_not_0"},   // 内部op.输入输出都是子图内部节点，写对应的key
      .outputs = {{"cast_0"}}
    };
    temp_nodes_.emplace_back(n5);

    const TempNode n6 = {
      .key = "cast_0",
      .op = "Cast",
      .inputs = {"logic_xor_0"},
      .outputs = {{"sum_0"}}
    };
    temp_nodes_.emplace_back(n6);

    const TempNode n7 = {
      .key = "sum_0",
      .op = "Sum",
      .inputs = {"cast_0","4"},
      .outputs = {{"neg_0"}}
    };
    temp_nodes_.emplace_back(n7);

    const TempNode n8 = {
      .key = "neg_0",
      .op = "Neg",
      .inputs = {"sum_0"},            // 上一个op（的key）
      .outputs = {{"0"}}              // 对接的下一个op.代表第n个外部输出
    };
    temp_nodes_.emplace_back(n8);     // 所有node都添加到该模板的节点集合中

    first_key_   = "greater_0";        // 该模板的第一个Node的key.  可以用来找该模板的第一个节点。如果原图中某节点的类型，和该节点的op类型能匹配上.就开始尝试match
    num_inputs_  = 5;                  // 外部输入op个数
    num_outputs_ = 1;                  // 我们要匹配的子图，最终的输出op

  }

  const string name() {
    return "logicsum_base";  // fused_op_
  }


  // 核心：定义匹配到子图后，如何把子图从原图中替换掉，替换成自己写的op
  // OptimizerFusionImpl会传入：
  // nodes: 匹配到的原始图中的所有节点
  // inputs，outputs:匹配到的原始图中的输入，输出
  bool add_subgraph(std::map<std::string, MatchedNode>& nodes,
    std::string name_prefix, Graph* g,
    std::vector<const Edge*>& inputs,
    std::vector<std::vector<const Edge*>>& outputs) override {
    // 对原始图中的输入，做自定义检查
    if (!CheckInputs(inputs)) {
      LOG(WARNING) << "Input check failed";
      return false;
    }

    // 可以自己写一个简单模板，替换模型的小部分节点。看看成不成功。匹配上会在这里打日志。或者自己加日志
    LOG(INFO) << "Fusion template[" << name() << "] match op[" << nodes[first_key_].node->name() <<
          "][new_name:" << name_prefix << "_" << name() << "]";

    // 把我们要新加入的op定义成一个节点，加入原始图中。 用我们自定义的类型,node.set_op("LogicalSum")。
    // 返回：已加入原始图的该节点
    Node* node_fused_logicsum = add_fused_logicsum_node(nodes, name_prefix, g, inputs, outputs);

    if (!node_fused_logicsum) {
      LOG(WARNING) << "Add node_fused_logicsum node failed";
      return false;
    }

    // 此时子图和我们的新op都在原始图中了。在原始图中，设计原始子图输入输出（对应边）的替换，替换成我们的op
    // 这样推理的时候，就不走原始子图了。直接用我们的op完成了替代
    return rebuild_graph(g, inputs, outputs, node_fused_logicsum);
  }

  // 对第一个，第三个输入边。的src节点做校验
  bool CheckConstZeroNode(const NodeDef& node_def) {
    Tensor val;
    Status s = GetNodeAttr(node_def, "value", &val);
    if (val.dtype() == DT_FLOAT) {
      auto v = val.flat<float>();
      for (int i = 0; i < v.size(); i++) {
        if (fabs(v(i)) > 1e-6) {
          return false;
        }
      }
      return true;
    }

    return false;
  }

  // 自己定义的对输入节点的检查（不是overwrite的）
  bool CheckInputs(std::vector<const Edge*>& inputs) {
    if (inputs.size() > 4) {
      return CheckConstZeroNode(inputs[1]->src()->def()) && CheckConstZeroNode(inputs[3]->src()->def());// 要求1号，3号节点的值是0，才能去做fusion
    } else {
      return false;
    }
  }

  bool CheckDynamicInputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<const Edge*>& fused_op_inputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return false;
  } 

  bool CheckDynamicOutputs(
      const Node* node, const TempNode* temp_node, int dy_mode, 
      std::vector<std::vector<const Edge*>>& fused_op_outputs, 
      std::map<const std::string, TempNode>& temp_node_map,
      std::map<std::string, MatchedNode>& matched_node_map) override {
    return false;
  }


 protected:
  // 自定义的op, 作为一个op节点加入图中
  // 模板匹配到的子图，就用这个替换
  virtual Node* add_fused_logicsum_node(
      std::map<std::string, MatchedNode>& nodes,
      std::string name_prefix, Graph* g,
      std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs) {
    // construct fused_logicsum node
    NodeDef fused_logicsum_node;                                      // 定义一个新的op节点，是你要用来代替子图的
    add_input(fused_logicsum_node, inputs[0]);        
    add_input(fused_logicsum_node, inputs[2]);                        // 指定新节点的input op, 一般同模板的输入op.
    fused_logicsum_node.set_op("LogicalSum");                         // 这里指定新op的类型。是你自己自定义且注册过的。*****
    fused_logicsum_node.set_name(name_prefix + name());                        // fused_op_  +  logicsum_base。 该模板对应的新op的名字
    fused_logicsum_node.set_device(nodes["greater_0"].node->def().device());
    AttrValue dtype_attr;
    dtype_attr.set_type(DT_FLOAT);
    fused_logicsum_node.mutable_attr()->insert({"T", dtype_attr});
    // Add node。 此时指定好了该op node的输入，op类型(可能是自定义的)
    Status status;
    Node* node_fused_logicsum_node = g->AddNode(fused_logicsum_node, &status);    // 把这个新op，加到原始的图中
    if (status != Status::OK() || !node_fused_logicsum_node) {
      LOG(WARNING) << "Add fused_logicsum node failed: " << status.error_message();
      return NULL;
    }

    return node_fused_logicsum_node;// 返回的已加入原始图的该节点？还是加入了该节点的原始图
  }


  // inputs:原始子图中input节点
  // outputs:原始子图中的output节点
  // node_fused_logicsum_node:我们的新op节点，用来替换原始子图的输入输出边。 把边的另一半对接成我们的op节点，删掉图中原始边。完成子图替换

  // 原图中的子图还在。
  // 但把子图原来的外部输入边，输出边，由对接子图改成对接我们的new op (且删掉了原来这些边)
  // 所以即使子图中剩下部分还在， 网络不与这部分连接了。 被我们的op替换掉了
  virtual bool rebuild_graph(Graph* g, std::vector<const Edge*>& inputs,
      std::vector<std::vector<const Edge*>>& outputs,
      Node* node_fused_logicsum_node) {
    if (inputs.size() < 5 || outputs.size() > 1) {
      LOG(WARNING) << "Input size[" << inputs.size() << "] is less then 5 or output size["
          << outputs.size() << "] is more then 1";
      return false;
    }
    
    // 用原来输出边（ori_edges）的左节点，右边改对接我们新op的第index个输入（dst_input）

    add_iedge(g, node_fused_logicsum_node, 0,               inputs[0]);     // 在原始图中，把输入0对应的这条边，右边对接我们的新op (删掉原来的边)
    //             新op                   作为新op的第0个输入    原始边                                                       
    add_iedge(g, node_fused_logicsum_node, 1,               inputs[2]);     // 在原始图中，把输入2对应的这条边，右边也对接我们的新op (删掉原来的边)
    //             新op                   作为新op的第1个输入    原始边 

    // 此时完成了我们的输入的改造。原来子图的这2个输入，直接对接我们的新op  （TODO：input边 1,3还在子图里。对应的边没删掉，编译时会不会仍走这部分子图）

    //  用原来输出边的右节点，左边对接我们新op的第index个输出
    add_oedges(g, node_fused_logicsum_node, 0, outputs[0]);

    // 新增了新边后，删掉了原来的边
    // 实现了用我们的op，接管原来输入输出的目的
    // 因此调用addsubgraph后，原始图变成了新图
    return true;
  }

};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_GRAPH_TEMPLATE_LOGICSUM_BASE_H_
