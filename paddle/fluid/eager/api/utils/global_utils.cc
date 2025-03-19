// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/api/manual/eager_manual/nodes/nodes.h"

namespace egr {

Controller* Controller::controller_ = new Controller();
thread_local std::shared_ptr<paddle::imperative::Tracer> Controller::tracer_ =
    std::make_shared<paddle::imperative::Tracer>();

Controller& Controller::Instance() { return *controller_; }

void Controller::SetExpectedPlace(const paddle::platform::Place& place) {
  tracer_->SetExpectedPlace(place);
}

void Controller::SetUsePromote(bool use_promote) {
  tracer_->SetUsePromote(use_promote);
}
bool Controller::GetUsePromote() const { return tracer_->GetUsePromote(); }

bool Controller::UseLayoutAutoTune() {
  bool use_autotune = false;
#if defined(PADDLE_WITH_CUDA)
  auto place = tracer_->ExpectedPlace();
  bool is_gpu_place = paddle::platform::is_gpu_place(place);
  if (is_gpu_place) {
    use_autotune = tracer_->UseLayoutAutoTune();
  }
#endif
  return use_autotune;
}

  void Controller::ClearForceSequentialNodes() { force_sequential_nodes_.clear(); 
  std::cout << "ClearForceSequentialNodes" << std::endl;
  }
  void Controller::PushBackForceSequentialNodes(GradNodeBase* node) {
    force_sequential_nodes_.push_back(node);
    // if (dynamic_cast<SyncBatchNormGradNode*>(node)) {
    //   std::cout << "PushBackForceSequentialNodes " << node << " ,size=" << force_sequential_nodes_.size() << " shape=" << dynamic_cast<SyncBatchNormGradNode*>(node)->x_.get_intermidiate_tensor().dims() << " data=" << dynamic_cast<SyncBatchNormGradNode*>(node)->x_.get_intermidiate_tensor().data() << std::endl;
    // } else {
    //   std::cout << "PushBackForceSequentialNodes " << node << " ,size=" << force_sequential_nodes_.size() << std::endl;
    // }
    // std::cout << "PushBackForceSequentialNodes " << node << " ,size=" << force_sequential_nodes_.size() << std::endl;
  }
  void Controller::EraseForceSequentialNodes(GradNodeBase* node) {
    for (auto iter = force_sequential_nodes_.begin();
         iter != force_sequential_nodes_.end();
         ++iter) {
      if (*iter == node) {
        force_sequential_nodes_.erase(iter);
        return;
      }
    }
    std::cout << "EraseForceSequentialNodes " << node << " ,size=" << force_sequential_nodes_.size() << std::endl;
  }

}  // namespace egr
