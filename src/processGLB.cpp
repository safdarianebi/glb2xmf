#include <iostream>                       // For cerr, 
#include <variant>                        // Required for std::variant
#include <tiny_gltf.h>                    // For tinygltf::Model, tinygltf::Mesh, ...
#include <glm/glm.hpp>                    // For glm::vec3, glm::vec4, ...
#include <glm/gtc/matrix_transform.hpp>   // For transformations
#include <glm/gtc/quaternion.hpp>         // For quaternion operations
#include <glm/gtx/quaternion.hpp>         // For quaternion-related functions
#include <glm/gtc/type_ptr.hpp>           // Required for value_ptr
#include <H5Cpp.h>                        // For saving data in HDF5 file
#include "pugixml.hpp"                    // For XMF files
#include <algorithm>                      // For std::replace
#include <filesystem>                     // For creating directories
#include <numeric>                        // For std::accumulate

// Global variables
const std::string OBJECT_NAME = "MANTA_RAY";

// Function to load GLB file
tinygltf::Model loadGLB(const std::string& file_name){
    std::cout << "Loading GLB file..." << std::endl;

    // Loading the GLB file
    tinygltf::TinyGLTF loader;
    std::string err, warn;
    tinygltf::Model model;

    // Print errors
    if (!loader.LoadBinaryFromFile(&model, &err, &warn, file_name)) {
        std::cerr << "Failed to load GLB file: " << err << std::endl;
    }

    // Print warnings
    if (!warn.empty()) {
        std::cerr << "Warnings: " << warn << std::endl;
    }

    // Return GLB file model
    return model;
    
}

// Function to extract maximum time in GLB file animation
double getMaxTimeInAnimation(const tinygltf::Model& model){
    // Maximum time definition
    double maxTime = 0;

    // Animation of GLB file
    const auto& animation = model.animations[0];

    // Analyze GlB file data
    std::cout << "Analyzing GLB file data...\n" << std::endl;
    std::cout << "\tNumber of animations: " << model.animations.size() << std::endl;
    std::cout << "\tAnimation Name: " << animation.name << std::endl;
    std::cout << "\tNumber of scenes in model: " << model.scenes.size() << std::endl;
    std::cout << "Mesh weigts: " << model.meshes[0].weights[0] << ", " << model.meshes[0].weights[1] << ", " << model.meshes[0].weights[2] << std::endl;
    

    // for (size_t i = 0; i < model.scenes.size(); ++i) {
    //     std::cout << "\tScene " << i << " has " << model.scenes[i].nodes.size() << " nodes." << std::endl;
    //     for (int nodeIndex : model.scenes[i].nodes) {
    //         std::cout << "\tRoot Node: " << nodeIndex << std::endl;
    //     }
    // }

    std::cout << "\tNumber of skins in model: " << model.skins.size() << std::endl;
    std::cout << "\tNumber of skins joints: " << model.skins[0].joints.size() << std::endl;
    std::cout << "\tFirst skin skeleton: " << model.skins[0].skeleton << std::endl;


    // for (const auto& joint: model.skins[0].joints) {
    //     std::cout << "\t" << joint;
    // }
    // std::cout << std::endl;

    // if ( true ) {
    //     std::cout << "\n\tTransformation matrix of node 0:" << std::endl;
    //     std::cout << "\t[";
    //     for (int r = 0; r < 4; ++r) {
    //         if (r > 0) std::cout << "\t ";

    //         for (int c = 0; c < 4; ++c) {
    //             float val = model.nodes[0].matrix[r * 4 + c];
            
    //             std::cout << std::setw(15) << std::setprecision(7) << std::scientific << float(val);
    //             if(c < 3) std::cout << "\t";
            
    //         }

    //         if (r < 3) std::cout << std::endl;
    //     }
    //     std::cout << "  ]";
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    std::cout << "\n\tNumber of Channels in animation: " << animation.channels.size() << std::endl;
    std::cout << "\tNumber of Nodes in model: " << model.nodes.size() << std::endl;

    // size_t count {}; // For counting channels

    // for (const auto& node: model.nodes) {
    //     std::cout << "\tnodeIndex: " << count++ << std::endl;
    //     std::cout << "\tnode.matrix: " << node.matrix.empty() <<  std::endl;
    //     std::cout << "\tnode.scale: " << node.scale.empty() << ", node.translation: " << node.translation.empty() << ", node.rotation: " << node.rotation.empty() << std::endl;
    // }

    // for (const auto& node: model.nodes){
    //     std::cout << "\n\tNode index: " << count++ << std::endl;
    //     std::cout << "\n\tNode mesh: " << node.mesh << std::endl;
    //     std::cout << "\tNode skin: " << node.skin << std::endl;
    //     std::cout << "\tNode camera: " << node.camera << std::endl;
    //     std::cout << "\tChilds:" << std::endl;
    //     for (const auto child: node.children) {
    //         std::cout << "\t\t" << child;
    //     }
    //     std::cout << std::endl;

    // }

    // Iterate through animation channels
    for (const auto& channel: animation.channels) {
        // Print channels data
        // std::cout << "\n\tChannel index: " << ++count << std::endl;
        // std::cout << "\tChannel target node: " << channel.target_node << std::endl;
        // std::cout << "\tChannel target path: " << channel.target_path << std::endl;

        // const tinygltf::Node& targetNode = model.nodes[channel.target_node];
        
        // std::cout << "\tChilds:" << std::endl;
        // for (const auto child: targetNode.children) {
        //     std::cout << "\t\t" << child;
        // }
        // std::cout << std::endl;

        // Access data in GLB file
        // Sampler of animation channel
        int samplerIndex = channel.sampler;
        const auto& sampler = animation.samplers[samplerIndex];

        // Accessor of animation channel
        int inputAccessorIndex = sampler.input;
        const auto& inputAccessor = model.accessors[inputAccessorIndex];

        // buffer 
        const tinygltf::BufferView& bufferView = model.bufferViews[inputAccessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

        // Iterate thorough time points
        for (size_t i = 0; i < inputAccessor.count; ++i) {
            size_t offset = bufferView.byteOffset + inputAccessor.byteOffset + i * sizeof(float);
            float timePoint = *reinterpret_cast<const float*>(&buffer.data[offset]);
            maxTime = std::max(maxTime, static_cast<double>(timePoint));
        }
    }

    // Analyze GLB file data
    std::cout << "\tMaximum time in animation: " << maxTime << std::endl;
    std::cout << "\tMesh primitives size: " << model.meshes[0].primitives.size() << std::endl;
    std::cout << "\tPrimitive mode: " << model.meshes[0].primitives[0].mode << std::endl;
    std::cout << "\tNumber of morph targets: " << model.meshes[0].primitives[0].targets.size() << std::endl;

    for (size_t i = 0; i < model.meshes[0].primitives[0].targets.size(); i++) {
        std::cout << "\n\t\tMorph Target " << i << " modifies: " << std::endl;
        for (const auto& attr : model.meshes[0].primitives[0].targets[i]) {
            std::cout << "\t\t  - " << attr.first << " (Accessor Index: " << attr.second << ")" << std::endl;
        }
    }

    std::cout << "\n\tAvailable attributes for first primitive:" << std::endl;
    for (const auto& attr : model.meshes[0].primitives[0].attributes) {
        std::cout << "\t\t- " << attr.first << std::endl;
    }

    std::cout << std::endl;
    std::cout << "GLB file Analyzed successfully!\n" << std::endl;

    // Return maximum time in animation
    return maxTime;
}

// Function to extract timepoints of animation channel
std::vector<float> extractTimePoints(int inputAccessorIndex, const tinygltf::Model& model){
    // Accessor of animation channel 
    const auto& inputAccessor = model.accessors[inputAccessorIndex];

    // Buffer 
    const tinygltf::BufferView& bufferView = model.bufferViews[inputAccessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    // Extract timpe points
    std::vector<float> timePoints(inputAccessor.count);
    size_t offset = bufferView.byteOffset + inputAccessor.byteOffset;
    memcpy(timePoints.data(), &buffer.data[offset], inputAccessor.count * sizeof(float));

    // Return timepoints of animation channel
    return timePoints;
}

// Function to find keyframe indices for interpolation
std::pair<int, int> findKeyframeIndices(const std::vector<float>& timePoints, double time){
    // For channel with one timepoint
    if (timePoints.size() == 1) return { 0, 0 };

    // Check if time is out of bounds of timepoints
    if ((time < timePoints.front()) || (time > timePoints.back())) {
        return { -1, -1 };
    }

    // Time is equall to first timepoint
    if (time == timePoints.front()) return { 0, 1};

    // Time is equal to last timepoint
    if (time == timePoints.back()) return { (int)timePoints.size() - 2, (int)timePoints.size() - 1 };

    // Iterate through timepoints to find indices
    for (size_t i = 0; i < timePoints.size(); ++i) {
        // Check if timepoint is larger than realted time
        if (timePoints[i] > time) return { (int)i - 1, (int)i };
    }

    // Return -1, -1 if time is not inside bounds
    return { -1, -1 };
}

// Function to compute interpolation factor
float computeInterpolationFactor(double time, const std::vector<float>& timePoints, int beforeIndex, int afterIndex) {
    return (time - timePoints[beforeIndex]) / (timePoints[afterIndex] - timePoints[beforeIndex]);
}

// Funtion to perform interpolation for transforamtions
std::variant<glm::vec3, glm::quat> interpolateTransformation(
    int outputAccessorIndex, const tinygltf::Model& model, int beforeIndex, int afterIndex, float t, const std::string& targetPath
) {
    // Acces buffer
    const auto& accessor = model.accessors[outputAccessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    // Define element size and transformation type
    // For translation transforation & scal transforation
    if (targetPath != "rotation") {
        // Size of glm::vec3 
        size_t elemSize = 3 * sizeof(float); 
        
        // Define transformation type for transforamtion & scale
        glm::vec3 beforeValue, afterValue;

        // Compute offset
        size_t stride = bufferView.byteStride? bufferView.byteStride: elemSize;
        size_t offsetBefore = bufferView.byteOffset + accessor.byteOffset + beforeIndex * stride;
        size_t offsetAfter = bufferView.byteOffset + accessor.byteOffset + afterIndex * stride;

        // Extract data from buffer
        memcpy(&beforeValue, &buffer.data[offsetBefore], elemSize);
        memcpy(&afterValue, &buffer.data[offsetAfter], elemSize);

        // Return interpolated translation
        return glm::mix(beforeValue, afterValue, t); 
    } 
    // For rotation transformation 
    else 
    {
        // Size of glm::quat 
        size_t elemSize = 4 * sizeof(float);  

        // Define transformation type for rotation
        glm::quat beforeValue, afterValue;

        // Compute offset
        size_t stride = bufferView.byteStride? bufferView.byteStride: elemSize;
        size_t offsetBefore = bufferView.byteOffset + accessor.byteOffset + beforeIndex * stride;
        size_t offsetAfter = bufferView.byteOffset + accessor.byteOffset + afterIndex * stride;

        // Extract data from buffer
        memcpy(&beforeValue, &buffer.data[offsetBefore], elemSize);
        memcpy(&afterValue, &buffer.data[offsetAfter], elemSize);
        
        // Return interpolated translation
        return glm::normalize(glm::slerp(beforeValue, afterValue, t));
    } 
}

// Function to access mesh attribute buffer, accessor, and bufferView
std::tuple<const tinygltf::Buffer&, const tinygltf::Accessor&, const tinygltf::BufferView&>
accessMeshAttribute(const tinygltf::Model& model, const std::string& attributeName) {
    std::cout << attributeName << " Atrribute:\n" << std::endl;

    // Find the attribute
    const auto it = model.meshes[0].primitives[0].attributes.find(attributeName);
    if (it == model.meshes[0].primitives[0].attributes.end()) {
        throw std::runtime_error("Attribute not found: " + attributeName);
    }

    // Get accessor and bufferView
    const tinygltf::Accessor& accessor = model.accessors[it->second];
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

    // Log component type
    std::cout << "\tComponent Type: ";
    switch (accessor.componentType) {
        case TINYGLTF_COMPONENT_TYPE_BYTE: std::cout << "BYTE"; break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: std::cout << "UNSIGNED_BYTE"; break;
        case TINYGLTF_COMPONENT_TYPE_SHORT: std::cout << "SHORT"; break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: std::cout << "UNSIGNED_SHORT"; break;
        case TINYGLTF_COMPONENT_TYPE_INT: std::cout << "INT"; break;
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: std::cout << "UNSIGNED_INT"; break;
        case TINYGLTF_COMPONENT_TYPE_FLOAT: std::cout << "FLOAT"; break;
        case TINYGLTF_COMPONENT_TYPE_DOUBLE: std::cout << "DOUBLE"; break;
        default: std::cout << "Unknown"; break;
    }
    std::cout << std::endl;

    // Log size of element
    std::cout << "\tElement Count per Vertex: " << accessor.type << std::endl;
    std::cout << "\tNumber of Vertices: " << accessor.count << std::endl;
    std::cout << "\tBuffer byte stride: " << bufferView.byteStride << std::endl;
    std::cout << "\tBuffer data size: " << buffer.data.size() << std::endl;
    std::cout << std::endl;

    // Return buffer, accessor, and bufferView as a tuple
    return std::forward_as_tuple(buffer, accessor, bufferView);
}

// Function to get interpolated morph weights
std::vector<float> interpolateMorphWeights(
    int outputAccessorIndex, const tinygltf::Model& model, int beforeIndex, int afterIndex, float t, size_t numMorphTargets
) {
    // Access the buffer
    const auto& accessor = model.accessors[outputAccessorIndex];
    const auto& bufferView = model.bufferViews[accessor.bufferView];
    const auto& buffer = model.buffers[bufferView.buffer];

    // Element size
    size_t elemSize = numMorphTargets * sizeof(float);

    // Use bufferView stride (if available)
    size_t stride = bufferView.byteStride > 0 ? bufferView.byteStride : elemSize;

    // Prepare data
    std::vector<float> beforeValue(numMorphTargets), afterValue(numMorphTargets);

    // Compute offsets
    size_t offsetBefore = bufferView.byteOffset + accessor.byteOffset + beforeIndex * stride;
    size_t offsetAfter = bufferView.byteOffset + accessor.byteOffset + afterIndex * stride;

    // Extract data from buffer
    memcpy(beforeValue.data(), &buffer.data[offsetBefore], elemSize);
    memcpy(afterValue.data(), &buffer.data[offsetAfter], elemSize);
    
    // std::cout << "weig bef: " << beforeValue[0] << ", " << beforeValue[1] << ", " << beforeValue[2] << std::endl;
    // std::cout << "weig aft: " << afterValue[0] << ", " << afterValue[1] << ", " << afterValue[2] << std::endl;

    if (beforeIndex == afterIndex) return beforeValue;
    
    // Interpolate
    std::vector<float> interpolatedMorphWeights(numMorphTargets);
    for (size_t i = 0; i < numMorphTargets; ++i) {
        interpolatedMorphWeights[i] = beforeValue[i] + (afterValue[i] - beforeValue[i]) * t;
    }

    // std::cout << "interpolate: " << interpolatedMorphWeights[0] << ", " << interpolatedMorphWeights[1] << ", " << interpolatedMorphWeights[2] << std::endl;

    return interpolatedMorphWeights;
}

// Function to get global transform
void getGlobalTransforms(const tinygltf::Model& model, int nodeIndex, glm::mat4& parentTransform, double time, std::vector<glm::mat4>& globalTransforms) {
    // std::cout << "\tnodeIndex: " << nodeIndex << std::endl;
    // Model animation
    const auto& animation = model.animations[0];

    // Grab node for node index
    const tinygltf::Node& node = model.nodes[nodeIndex];

    // Flag to check if the node is animated
    bool animated_scale = false;
    bool animated_translation = false;
    bool animated_rotation = false;

    // std::cout << "\tnodeIndex: " << nodeIndex << " nodeName: " << node.name << std::endl;
    // std::cout << "\tnode childs: " << std::endl;
    // std::cout << "\t" << std::endl;
    // for (const auto& child: node.children) {
    //     std::cout << "\t" << child << "\t" << std::endl;
    // }
    // std::cout << "\t" << std::endl;

    // Grab transformation matrix at time
    // Initialize transformations with default values
    glm::vec3 translation(0.0f);
    glm::vec3 scale(1.0f, 1.0f, 1.0f);
    glm::quat rotation(1.0f, 0.0f, 0.0f, 0.0f);

    for (const auto& channel: animation.channels) {
        // Check if channel affects the present node
        if (channel.target_node == nodeIndex) {
            if (channel.target_path != "weights") {
                if (channel.target_path == "scale") animated_scale = true;
                if (channel.target_path == "translation") animated_translation = true;
                if (channel.target_path == "rotation") animated_rotation = true;
                std::cout << "\tNode " << nodeIndex << " " << channel.target_path << " is animated!" << std::endl;
            } 

            // Sampler
            int samplerIndex = channel.sampler;
            const auto& sampler = animation.samplers[samplerIndex];

            // Find keyfram indices
            std::vector<float> timePoints = extractTimePoints(sampler.input, model);

            auto [beforeIndex, afterIndex] = findKeyframeIndices(timePoints, time);

            // std::cout << "\tnodeIndex: " << nodeIndex << ", before: " << timePoints[beforeIndex] << ", time: " << time<< ", after: " << timePoints[afterIndex] << std::endl;

            float t = computeInterpolationFactor(time, timePoints, beforeIndex, afterIndex);

            // Check channel target path
            if (channel.target_path == "translation") 
                // Get interpolated translation 
                translation = std::get<glm::vec3>(interpolateTransformation(sampler.output, model, beforeIndex, afterIndex, t, channel.target_path));

            else if (channel.target_path == "scale")
                // Get interpolated scale 
                scale = std::get<glm::vec3>(interpolateTransformation(sampler.output, model, beforeIndex, afterIndex, t, channel.target_path));

            else if (channel.target_path == "rotation")
                // Get interpolated rotation 
                rotation = std::get<glm::quat>(interpolateTransformation(sampler.output, model, beforeIndex, afterIndex, t, channel.target_path));
            
            // Channel target path would be morph, but no need to do anything
        }
    }

    bool animated = animated_rotation || animated_scale || animated_translation;
    if (!animated) std::cout << "\tNode " << nodeIndex << " has no animated transformation" << std::endl;

    // Compute the local transformation matrix
    // if ((!animated) && (node.matrix.empty())) {
    //     if ((!node.scale.empty()) && (!node.translation.empty()) && (!node.rotation.empty())) {
    //         std::cout << "\tNo transformation in animation, using node trasformations (scale, translation, and rotation) for node " << nodeIndex << "!" << std::endl;
    //         scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
    //         translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
    //         rotation = glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]);
    //     }
    // }

    // if ((!animated) && (node.matrix.empty())) {

    //     if(!node.scale.empty()) {
    //         std::cout << "\tNo transformation in animation, using node scale for node " << nodeIndex << "!" << std::endl;
    //         scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
    //     }
    //     else if(!node.translation.empty()) {
    //         std::cout << "\tNo transformation in animation, using node translation for node " << nodeIndex << "!" << std::endl;
    //         translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
    //     }
        
    //     else if(!node.rotation.empty()) {
    //         std::cout << "\tNo transformation in animation, using node rotation for node " << nodeIndex << "!" << std::endl;
    //         rotation = glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]);
    //     }
    //     else std::cout << "\tNode " << nodeIndex << " local transformation matrix is identity matrix!" << std::endl;
        
    // }

    if ((node.matrix.empty())) {

        if((!node.scale.empty()) && (!animated_scale)) {
            std::cout << "\tNo transformation in animation, using node scale for node " << nodeIndex << "!" << std::endl;
            scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
        }
        else if((!node.translation.empty()) && (!animated_translation)) {
            std::cout << "\tNo transformation in animation, using node translation for node " << nodeIndex << "!" << std::endl;
            translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
        }
        
        else if((!node.rotation.empty()) && (!animated_rotation)) {
            std::cout << "\tNo transformation in animation, using node rotation for node " << nodeIndex << "!" << std::endl;
            rotation = glm::quat(node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3]);
        }
        else std::cout << "\tNode " << nodeIndex << " local transformation matrix is identity matrix!" << std::endl;
        
    }

    glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    glm::mat4 R = glm::toMat4(rotation);

    // If node has transformations
    glm::mat4 localTransform = T * R * S;

    // If node doesn't have any transformation in animation
    if ((!animated) && (!node.matrix.empty())) {
        std::cout << "\tNo transformation in animation, using node matrix for node " << nodeIndex << "!" << std::endl;
        localTransform = glm::make_mat4(node.matrix.data());
    }

    // Compute global trnasformation matrix
    glm::mat4 globalTransform = parentTransform * localTransform;
    globalTransforms[nodeIndex] = globalTransform;

    if (node.name == "manta_ray_armature") {
        std::cout << "\tarmature node: " << node.name << ", nodeIndex: " << nodeIndex << std::endl;
        globalTransforms[nodeIndex] = parentTransform;
    } 

    // Print a sample
    // if (nodeIndex % 1 == 0) {
    //     std::cout << "\n\tGlobal transformation matrix of node " << nodeIndex << ":" << std::endl;
    //     std::cout << "\t[";
    //     for (int r = 0; r < 4; ++r) {
    //             if (r > 0) std::cout << "\t ";
            
    //         for (int c = 0; c < 4; ++c) {
    //             float val = globalTransforms[nodeIndex][r][c];
                        
    //             std::cout << std::setw(15) << std::setprecision(7) << std::scientific << float(val);
    //             if(c < 3) std::cout << "\t";
                        
    //         }
            
    //         if (r < 3) std::cout << std::endl;
    //     }
    //     std::cout << "  ]\n";
    //     std::cout << std::endl;
    // }
    
    // Apply transformaion to node childs
    for (const auto& child: node.children) {
        getGlobalTransforms(model, child, globalTransform, time, globalTransforms);
    }
}

// Function to get inversed bind matrixes
void getInversedBinds(const tinygltf::Model& model, std::vector<glm::mat4>& inversedBinds) {
    // Access buffer
    const tinygltf::Accessor &accessor_ibm = model.accessors[model.skins[0].inverseBindMatrices];
    const tinygltf::BufferView &bufferView_ibm = model.bufferViews[accessor_ibm.bufferView];
    const tinygltf::Buffer &buffer_ibm = model.buffers[bufferView_ibm.buffer];

    // Loop thorough skin joints
    for (size_t i = 0; i < model.skins[0].joints.size(); ++i) {
        // Copy the inversed bind matrix data from the buffer
        size_t matSize = 16 * sizeof(float);
        size_t stride_ibm = bufferView_ibm.byteStride? bufferView_ibm.byteStride: matSize;
        size_t offset_ibm = bufferView_ibm.byteOffset + accessor_ibm.byteOffset + i * stride_ibm;
        std::vector<float> matrixData(16);
        memcpy(matrixData.data(), &buffer_ibm.data[offset_ibm], matSize);

        // Copy matrix data into a glm::mat4
        glm::mat4 inversedBindMatrix;
        memcpy(glm::value_ptr(inversedBindMatrix), matrixData.data(), sizeof(glm::mat4));

        // if (glm::length(glm::vec3(inversedBindMatrix[0][0], inversedBindMatrix[1][1], inversedBindMatrix[2][2])) > 10.0f) { // Detect extreme scaling
        //     std::cout << "\tExtreme scaling for node " << model.skins[0].joints[i] << std::endl;
        //     inversedBindMatrix = inversedBindMatrix * 0.01f; // Compensate for 100x scale
        // }

        inversedBinds[i] = inversedBindMatrix;

        if (model.nodes[model.skins[0].joints[i]].name == "manta_ray_armature") inversedBinds[i] = glm::mat4(0.0f);

        // Print a sample
        // if (i % 1 == 0) {
        //     std::cout << "\n\tInversed bind matrix of node " << model.skins[0].joints[i] << ":" << std::endl;
        //     std::cout << "\t[";
        //     for (int r = 0; r < 4; ++r) {
        //             if (r > 0) std::cout << "\t ";
                
        //         for (int c = 0; c < 4; ++c) {
        //             float val = inversedBinds[i][r][c];
                            
        //             std::cout << std::setw(15) << std::setprecision(7) << std::scientific << float(val);
        //             if(c < 3) std::cout << "\t";
                            
        //         }
                
        //         if (r < 3) std::cout << std::endl;
        //     }
        //     std::cout << "  ]\n";
        //     std::cout << std::endl;
        // }
    }
}   

// Function to find largest morph vertex 
glm::vec3 findMaxMorphDelta(
    const tinygltf::Accessor& accessor,
    const tinygltf::BufferView& bufferView,
    const tinygltf::Buffer& buffer
) {
    
    const size_t vertexCount = accessor.count;
    const size_t vertexSize = 3 * sizeof(float);
    const size_t stride = bufferView.byteStride ? bufferView.byteStride : vertexSize;

    glm::vec3 maxDelta(0.0f);
    float maxMagnitude = 0.0f;

    float maxDelta0 = 0.0f;
    float maxDelta1 = 0.0f;
    float maxDelta2 = 0.0f;

    float minDelta0 = 0.0f;
    float minDelta1 = 0.0f;
    float minDelta2 = 0.0f;


    for (size_t i = 0; i < vertexCount; ++i) {
        glm::vec3 morphVertex;
        size_t offset = bufferView.byteOffset + accessor.byteOffset + i * stride;

        memcpy(&morphVertex, &buffer.data[offset], vertexSize);
        
        float currentMagnitude = glm::length(morphVertex);
        if (currentMagnitude > maxMagnitude) {
            maxMagnitude = currentMagnitude;
            maxDelta = morphVertex;
        }

        if (morphVertex[0] > maxDelta0) maxDelta0 = morphVertex[0];
        if (morphVertex[1] > maxDelta1) maxDelta1 = morphVertex[1];
        if (morphVertex[2] > maxDelta2) maxDelta2 = morphVertex[2];

        if (morphVertex[0] > maxDelta0) maxDelta0 = morphVertex[0];
        if (morphVertex[1] > maxDelta1) maxDelta1 = morphVertex[1];
        if (morphVertex[2] > maxDelta2) maxDelta2 = morphVertex[2];

    }

    float maxLen0 = (maxDelta0 - minDelta0)? (maxDelta0 - minDelta0): 1.0f;
    float maxLen1 = (maxDelta1 - minDelta1)? (maxDelta1 - minDelta1): 1.0f;
    float maxLen2 = (maxDelta2 - minDelta2)? (maxDelta2 - minDelta2): 1.0f;

    return glm::vec3(maxLen0, maxLen1, maxLen2);
}

// Function to apply morph
void applyMorph(
    const tinygltf::Model& model,
    tinygltf::Model& transformedModel,
    const std::tuple<const tinygltf::Buffer&, const tinygltf::Accessor&, const tinygltf::BufferView&>& posTuple,
    std::tuple<tinygltf::Buffer&, tinygltf::Accessor&, tinygltf::BufferView&>& transformedPosTuple,
    std::tuple<tinygltf::Buffer&, tinygltf::Accessor&, tinygltf::BufferView&>& transformedNormalTuple,
    double time
) {

    // Unpack position tuples
    const auto& [buffer_pos, accessor_pos, bufferView_pos] = posTuple;
    auto& [buffer_pos_t, accessor_pos_t, bufferView_pos_t] = transformedPosTuple;

    // Number of vertexes
    size_t numVertexes = accessor_pos_t.count;

    // Size of vertex
    size_t vertexSize = 3 * sizeof(float);

    // Model animation
    const auto& animation = model.animations[0];
    
    // Animation channel & sampler
    int samplerIndex = 0;
    int nodeIndex = 0;

    for (const auto& channel: animation.channels) {
        if (channel.target_path == "weights") {
            samplerIndex = channel.sampler;
            nodeIndex = channel.target_node;
            break;
        }
    }

    const auto& sampler = animation.samplers[samplerIndex];
    // std::cout << "\tsampler interpolation: " << sampler.interpolation << std::endl;

    // Find keyfram indices
    std::vector<float> timePoints = extractTimePoints(sampler.input, model);

    auto [beforeIndex, afterIndex] = findKeyframeIndices(timePoints, time);

    float t = computeInterpolationFactor(time, timePoints, beforeIndex, afterIndex);

    std::cout << "\tApplying morph to node " << nodeIndex << "...\n" << std::endl;

    // Apply morph
    // Number of morph targets
    size_t numMorphTargets = model.meshes[0].primitives[0].targets.size();

    // Get interpolated morph weights
    std::vector<float> morphWeights = interpolateMorphWeights(sampler.output, model, beforeIndex, afterIndex, t, numMorphTargets);

    // std::cout << "\tNumber of morph targets: " << numMorphTargets << std::endl;
    // std::cout << "\tMorph Weights: ";
    // for (float w : morphWeights) std::cout << w << " ";
    // std::cout << std::endl;

    // Applying morph to mesh
    for (size_t targetIndex = 0; targetIndex < numMorphTargets; ++targetIndex) {
        // Get morph target
        const auto& target = model.meshes[0].primitives[0].targets[targetIndex];
        
        // Access weight for morph target
        float weight = morphWeights[targetIndex];
        if (weight == 0.0f) continue;

        if (std::isnan(weight)) {
            std::cerr << "Error: NaN detected in morph weight for target " << targetIndex << std::endl;
            return;
        }

        // Applying morph to morph attributes
        for (const auto& attribute: target) {
            std::cout << "\tMorph target: " << attribute.first << ", Morph weight: " << weight << "\n" << std::endl;

            // Counter for printing vertexes
            size_t count = 0;

            // Access morph target data
            const tinygltf::Accessor& accessor_morph = model.accessors[attribute.second];
            const tinygltf::BufferView& bufferView_morph = model.bufferViews[accessor_morph.bufferView];
            const tinygltf::Buffer& buffer_morph = model.buffers[bufferView_morph.buffer];

            // Find largest morph vertex
            glm::vec3 maxMorphVertex = findMaxMorphDelta(accessor_morph, bufferView_morph, buffer_morph);

            // Access attribute
            auto& buffer_atr = buffer_pos_t;
            auto& accessor_atr = accessor_pos_t;
            auto& bufferView_atr = bufferView_pos_t;

            if (attribute.first == "NORMAL") {
                auto& [buffer_atr, accessor_atr, bufferView_atr] = transformedNormalTuple;
            }
            else if (attribute.first != "POSITION") {
                auto it_atr = transformedModel.meshes[0].primitives[0].attributes.find(attribute.first);
                if (it_atr == transformedModel.meshes[0].primitives[0].attributes.end()) {
                    throw std::runtime_error("Attribute not found: " + attribute.first);
                }
                accessor_atr = transformedModel.accessors[it_atr->second];
                bufferView_atr = transformedModel.bufferViews[accessor_atr.bufferView];
                buffer_atr = transformedModel.buffers[bufferView_atr.buffer];
            }

            // Applying morph to attribute vertices
            for (size_t i = 0; i < numVertexes; ++i) {
                // Compute morph offset
                size_t stride_morph = bufferView_morph.byteStride? bufferView_morph.byteStride: 3 * sizeof(float);
                size_t morphOffset = bufferView_morph.byteOffset + accessor_morph.byteOffset + i * stride_morph;

                if (morphOffset >= buffer_morph.data.size()) {
                    std::cerr << "Error: Morph offset out of bounds: " << morphOffset << std::endl;
                    continue;
                }

                // Read vertex of morph
                glm::vec3 morphVertex;
                memcpy(&morphVertex, &buffer_morph.data[morphOffset], vertexSize);
                if (glm::any(glm::isnan(morphVertex))) {
                    std::cerr << "Error: NaN detected in morph vertex at index " << i << std::endl;
                    return;
                }

                // Compute attribute offset 
                size_t stride_atr =  bufferView_atr.byteStride?  bufferView_atr.byteStride: 3 * sizeof(float);
                size_t atrOffset = bufferView_atr.byteOffset + accessor_atr.byteOffset + i * stride_atr;
                
                if (atrOffset >= buffer_atr.data.size()) {
                    std::cerr << "Error: Attribute offset out of bounds: " << atrOffset << std::endl;
                    continue;
                }

                // Read vertex position
                glm::vec3 transformedVertex;
                memcpy(&transformedVertex, &buffer_atr.data[atrOffset], vertexSize);
                
                if (glm::any(glm::isnan(transformedVertex))) {
                    std::cerr << "Error: NaN detected in transformed vertex before morphing at index " << i << std::endl;
                    return;
                }

                // Apply morph
                glm::vec3 vertex;
                memcpy(&vertex, &buffer_pos.data[atrOffset], vertexSize);
                
                if ((vertex[0] > 0.0f) && (vertex[1] > 0.2f)) transformedVertex += 0.0005f * weight * morphVertex;
                else transformedVertex += 0.0005f * weight * morphVertex;

                if (glm::any(glm::isnan(transformedVertex))) {
                    std::cerr << "Error: NaN detected in transformed vertex after morphing at index " << i << std::endl;
                    return;
                }

                // Store the transformed vertex (convert vec4 → vec3)
                memcpy(&buffer_atr.data[atrOffset], &transformedVertex, vertexSize);

                if ((count < 100) && (weight != 0) && (i % 3000 == 0) && (attribute.first == "POSITION")) {
                    std::cout << "\tVertex index: " << i << std::endl;
                    std::cout << "\tMorph Vertex: ( " << morphVertex[0] << ", " << morphVertex[1] << ", " << morphVertex[2] << " )" << std::endl;
                    std::cout << "\tVertex: ( " << vertex[0] << ", " << vertex[1] << ", " << vertex[2] << " )" << std::endl;
                    std::cout << "\tTransformed vertex: ( " << transformedVertex[0] << ", " << transformedVertex[1] << ", " << transformedVertex[2] << " )\n" << std::endl;
                    ++count;
                }
                
            }

            if (weight != 0)
            std::cout << "\t.\n\t.\n\t.\n\n" << std::endl;
        }
    }

    std::cout << "\tSuccessfully apllied morph to node " << nodeIndex << "!\n" << std::endl;
}

// Functin to apply transformation to nodes
void applyTransformationToNodes(
    const tinygltf::Model& model,
    tinygltf::Model& transformedModel,
    const std::tuple<const tinygltf::Buffer&, const tinygltf::Accessor&, const tinygltf::BufferView&>& posTuple,
    const std::tuple<const tinygltf::Buffer&, const tinygltf::Accessor&, const tinygltf::BufferView&>& jointTuple,
    const std::tuple<const tinygltf::Buffer&, const tinygltf::Accessor&, const tinygltf::BufferView&>& weightTuple,
    std::tuple<tinygltf::Buffer&, tinygltf::Accessor&, tinygltf::BufferView&>& transformedPosTuple,
    std::tuple<tinygltf::Buffer&, tinygltf::Accessor&, tinygltf::BufferView&>& transformedNormalTuple,
    glm::mat4& parentTransform,
    double time
) {
    // Counter for printing vertexes
    size_t count = 0;
    
    // Unpack attribute tuples
    const auto& [buffer_pos, accessor_pos, bufferView_pos] = posTuple;
    const auto& [buffer_joint, accessor_joint, bufferView_joint] = jointTuple;
    const auto& [buffer_weight, accessor_weight, bufferView_weight] = weightTuple;
    auto& [buffer_pos_t, accessor_pos_t, bufferView_pos_t] = transformedPosTuple;

    // Number of vertexes
    size_t numVertexes = accessor_pos.count;

    // Size of vertex
    size_t vertexSize = 3 * sizeof(float);

    // Aplly skinning to mesh
    std::cout << "\tApplying skinning to nodes " << model.skins[0].joints.front() << "-" << model.skins[0].joints.back() << "...\n" << std::endl; 

    // Get nodes global transformation matrixes
    std::vector<glm::mat4> globalNodeTransforms;
    globalNodeTransforms.resize(model.nodes.size(), glm::mat4(1.0f));
    getGlobalTransforms(model, 0, parentTransform, time, globalNodeTransforms);

    // Access inversed bind matrix
    std::vector<glm::mat4> inversedBinds;
    inversedBinds.resize(model.skins[0].joints.size(), glm::mat4(1.0f));
    getInversedBinds(model, inversedBinds);

    // Joints matrixes
    std::vector<int> jointIndices = model.skins[0].joints; 
    std::vector<glm::mat4> jointMatrixes;
    jointMatrixes.resize(jointIndices.size(), glm::mat4(1.0f));

    int nodeIndex{};
    for (const auto& channel: model.animations[0].channels) {
        if (channel.target_path == "weights")
            nodeIndex = channel.target_node;
    }

    int skeleton = model.skins[0].skeleton;

    for (size_t i = 0; i < jointIndices.size(); ++i) {
        jointMatrixes[i] =  globalNodeTransforms[jointIndices[i]] * inversedBinds[i];
        // std::cout << "\tnodeIndex: " << jointIndices[i] << " jointIndex: " << i << std::endl;
    }

    // Print a sample joint matrix
    // int jointIndex{};
    // for (auto& joint:jointMatrixes) {
    //     std::cout << "\n\tJoint matrix of node " << jointIndices[jointIndex++] << ":" << std::endl;
    //     std::cout << "\t[";
    //     for (int r = 0; r < 4; ++r) {
    //             if (r > 0) std::cout << "\t ";
        
    //         for (int c = 0; c < 4; ++c) {
    //             float val = joint[r][c];
                    
    //             std::cout << std::setw(15) << std::setprecision(7) << std::scientific << float(val);
    //             if(c < 3) std::cout << "\t";
                    
    //         }
        
    //         if (r < 3) std::cout << std::endl;
    //     }
    //     std::cout << "  ]\n";
    //     std::cout << std::endl;
    // }

    
    // Loop through all vertices
    for (size_t i = 0; i < numVertexes; ++i) {
        // Read joint
        size_t stride_joint = bufferView_joint.byteStride? bufferView_joint.byteStride: 4 * sizeof(uint16_t);
        size_t offset_joint = bufferView_joint.byteOffset + accessor_joint.byteOffset +  i * stride_joint;
        std::vector<uint16_t> joints(4);
        memcpy(joints.data(), &buffer_joint.data[offset_joint], 4 * sizeof(uint16_t));
        // if (i % 3000 == 0)
        // std::cout << "joint: (" << joints[0] << ", " << joints[1] << ", " << joints[2] << ", " << joints[3] << ")" << std::endl;

        // Access weights
        size_t stride_weight = bufferView_weight.byteStride? bufferView_weight.byteStride: 4 * sizeof(float);
        size_t offset_weight = bufferView_weight.byteOffset + accessor_weight.byteOffset + i * stride_weight;
        std::vector<float> weights(4);
        memcpy(weights.data(), &buffer_weight.data[offset_weight], 4 * sizeof(float));
        // float sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        // if (i % 3000 == 0)
        // std::cout << "weights: (" << weights[0] << ", " << weights[1] << ", " << weights[2] << ", " << weights[3] << ")" << std::endl;

        // Compute offset 
        size_t stride_pos_t = bufferView_pos_t.byteStride? bufferView_pos_t.byteStride: 3 * sizeof(float);
        size_t offset = bufferView_pos_t.byteOffset + accessor_pos_t.byteOffset + i * stride_pos_t;

        // Read vertex position
        glm::vec3 vertex;
        memcpy(&vertex, &buffer_pos.data[offset], vertexSize);

        // Read transformed vertex posiotion
        glm::vec3 transformedVertex;
        memcpy(&transformedVertex, &buffer_pos_t.data[offset], vertexSize);


        // skin matrix 
        glm::mat4 skinMatrix = glm::inverse(globalNodeTransforms[nodeIndex]) * (
            weights[0] * jointMatrixes[joints[0]] + 
            weights[1] * jointMatrixes[joints[1]] +
            weights[2] * jointMatrixes[joints[2]] +
            weights[3] * jointMatrixes[joints[3]] 
        );

        // Apply transfomration
        glm::vec4 transformedVertexMat = skinMatrix * glm::vec4(transformedVertex, 1.0f);

        // Store the transformed vertex (convert vec4 → vec3)
        transformedVertex = glm::vec3(transformedVertexMat);
        memcpy(&buffer_pos_t.data[offset], &transformedVertex, vertexSize);

        if ((count < 3) && (i % 3000 == 0)) {
            std::cout << "\tVertex index: " << i << std::endl;
            std::cout << "\tVertex: ( " << vertex[0] << ", " << vertex[1] << ", " << vertex[2] << " )" << std::endl;
            std::cout << "\tTrsformed vertex: ( " << transformedVertex[0] << ", " << transformedVertex[1] << ", " << transformedVertex[2] << " )\n" << std::endl;
            ++count;
        }    
    }
    
    std::cout << "\t.\n\t.\n\t.\n\n\tSuccessfully apllied skinning to nodes " << model.skins[0].joints.front() << "-" << model.skins[0].joints.back() << "!\n" << std::endl; 
}

// Function to save data in HDF5 file
template <typename T>
void saveToHDF5(H5::H5File& file, const std::vector<T>& data, const std::string& datasetName, hsize_t dim1, hsize_t dim2){
    try {
        std::cout << "\tSaving dataset '" << datasetName << "' with size: " << data.size() << " in HDF5 file..." << std::endl;
        
        // Dataspace
        hsize_t dims[2] = {dim1, dim2};
        H5::DataSpace dataspace(2, dims);

        // Data type
        H5::PredType hdf5Type = (std::is_same<T, float>::value) ? H5::PredType::NATIVE_FLOAT
            : (std::is_same<T, int>::value) ? H5::PredType::NATIVE_INT
            : throw std::runtime_error("Unsupported data type for HDF5 output.");

        // Write data in HDF5 file
        H5::DataSet dataset = file.createDataSet(datasetName, hdf5Type, dataspace);
        dataset.write(data.data(), hdf5Type);

        std::cout << "\tSuccessfully saved dataset '" << datasetName << "' in HDF5 file!\n" << std::endl;
    }
    catch (H5::Exception& e) {
        std::cerr << "\tHDF5 error: " << e.getDetailMsg() << std::endl;
    }
}

// Function to format timestep
std::string formatTimestep(int timestep) {
    char buffer[20];
    snprintf(buffer, sizeof(buffer), "%012d", timestep);
    return std::string(buffer);
}

// Function to save data in .xmf files
bool saveToXMF(const std::string& hdf5Filename, const std::string&xmfFilename, int numVertices, int numTriangles) {
    std::cout << "\tCreating XMF file..." << std::endl;

    // Correct HDF5 file name for xmf file
    std::string correctedHDF5Path = hdf5Filename;
    size_t pos = correctedHDF5Path.find("output/");
    if (pos != std::string::npos) {
        correctedHDF5Path.replace(pos, 7, "./");
    }

    // Create xml document
    pugi::xml_document doc;

    // Add DOCTYPE for Xdmf compatibility
    pugi::xml_node doctype = doc.append_child(pugi::node_doctype);
    doctype.set_value("Xdmf SYSTEM \"Xdmf.dtd\" []");

    pugi::xml_node xmf = doc.append_child("Xdmf");
    xmf.append_attribute("xmlns:xi") = "http://www.w3.org/2003/XInclude";
    xmf.append_attribute("Version") = "2.2";

    // Domain
    pugi::xml_node domain = xmf.append_child("Domain");
    pugi::xml_node collectionGrid = domain.append_child("Grid");
    collectionGrid.append_attribute("Name") = "Domain";
    collectionGrid.append_attribute("GridType") = "Collection";

    // Grid
    pugi::xml_node grid = collectionGrid.append_child("Grid");
    grid.append_attribute("Name") = "Subdomain 0 0";
    grid.append_attribute("GridType") = "Uniform";

    // Topology (Triangle Mesh)
    pugi::xml_node topology = grid.append_child("Topology");
    topology.append_attribute("TopologyType") = "Triangle";
    topology.append_attribute("NumberOfElements") = numTriangles;

    pugi::xml_node topologyData = topology.append_child("DataItem");
    topologyData.append_attribute("Dimensions") = (std::to_string(numTriangles) + " 3").c_str();
    topologyData.append_attribute("Format") = "HDF";
    topologyData.text().set((correctedHDF5Path + ":/Triangles").c_str());

    // Geometry (Vertex Positions)
    pugi::xml_node geometry = grid.append_child("Geometry");
    geometry.append_attribute("GeometryType") = "XYZ";

    pugi::xml_node positionData = geometry.append_child("DataItem");
    positionData.append_attribute("Dimensions") = (std::to_string(numVertices) + " 3").c_str();
    positionData.append_attribute("Format") = "HDF";
    positionData.text().set((correctedHDF5Path + ":/Position").c_str());

    // Attribute (Position)
    pugi::xml_node attribute = grid.append_child("Attribute");
    attribute.append_attribute("Name") = "Position";
    attribute.append_attribute("AttributeType") = "Vector";

    pugi::xml_node attributeData = attribute.append_child("DataItem");
    attributeData.append_attribute("Dimensions") = (std::to_string(numVertices) + " 3").c_str();
    attributeData.append_attribute("Format") = "HDF";
    attributeData.text().set((correctedHDF5Path + ":/Position").c_str());

    // Save XMF file
    bool saved_file = doc.save_file(xmfFilename.c_str(), " ");
    std::cout << "\tSuccessfully created XMF file!\n" << std::endl;

    return saved_file;
}

// Function that processes GLB file
void processGLB(const std::string& file_name){
    // Load GLB file
    const tinygltf::Model& model = loadGLB(file_name);

    // Check model meshes
    if (model.meshes.empty()) {
        std::cerr << "No meshes found in GLB file!" << std::endl;
        return;
    }

    std::cout << "GLB file loaded successfully!\n" << std::endl;

    // Analyze data in GLB file
    double maxTime = getMaxTimeInAnimation(model); // Maximum time in animation
    
    // Access mesh attributes buffer
    const auto& [buffer_pos, accessor_pos, bufferView_pos] = accessMeshAttribute(model, "POSITION");
    const auto& [buffer_joint, accessor_joint, bufferView_joint] = accessMeshAttribute(model, "JOINTS_0");
    const auto& [buffer_weight, accessor_weight, bufferView_weight] = accessMeshAttribute(model, "WEIGHTS_0");
    const auto& [buffer_normal, accessor_normal, bufferView_normal] = accessMeshAttribute(model, "NORMAL");


    // Creat mesh attribtes tupples
    auto posTuple = std::forward_as_tuple(buffer_pos, accessor_pos, bufferView_pos);
    auto jointTuple = std::forward_as_tuple(buffer_joint, accessor_joint, bufferView_joint);
    auto weightTuple = std::forward_as_tuple(buffer_weight, accessor_weight, bufferView_weight);

    // Number of vertices
    size_t numVertices = accessor_pos.count;
    size_t vertexSize = 3 * sizeof(float);

    // Define number of timesteps:
    int numberOfTimeSteps = 1000;
    double dt = maxTime / numberOfTimeSteps;
    // Iterate through timesteps
    for (size_t timeStep = 0; timeStep < 500; ++timeStep) {
        // Time definition
        double time = timeStep * dt;

        // Define transformed model
        tinygltf::Model transformedModel = model;
    
        // Access transfomed mesh postions
        auto it_pos_t = transformedModel.meshes[0].primitives[0].attributes.find("POSITION");
        tinygltf::Accessor& accessor_pos_t = transformedModel.accessors[it_pos_t->second];
        tinygltf::BufferView& bufferView_pos_t = transformedModel.bufferViews[accessor_pos_t.bufferView];
        tinygltf::Buffer& buffer_pos_t = transformedModel.buffers[bufferView_pos_t.buffer];

        // Access transfomed mesh normal
        auto it_normal_t = transformedModel.meshes[0].primitives[0].attributes.find("NORMAL");
        tinygltf::Accessor& accessor_normal_t = transformedModel.accessors[it_normal_t->second];
        tinygltf::BufferView& bufferView_normal_t = transformedModel.bufferViews[accessor_normal_t.bufferView];
        tinygltf::Buffer& buffer_normal_t = transformedModel.buffers[bufferView_normal_t.buffer];    

        // Create transformed position and normal tuples
        auto transformedPosTuple = std::forward_as_tuple(buffer_pos_t, accessor_pos_t, bufferView_pos_t);
        auto transformedNormalTuple = std::forward_as_tuple(buffer_normal_t, accessor_normal_t, bufferView_normal_t);
    
        // Get mesh at related timestep        
        std::cout << "Getting mesh at "<< time << " s" << "...\n"<< std::endl;

        // Get glolbal transform 
        glm::mat4 parentTransform = glm::mat4(1.0f);

        // // Apply morph to the mesh
        // applyMorph(model, transformedModel, 
        //            posTuple, transformedPosTuple, transformedNormalTuple, time);

        // Apply transformation to nodes recursively
        applyTransformationToNodes(model, transformedModel, 
                                   posTuple, jointTuple, weightTuple,
                                   transformedPosTuple, transformedNormalTuple,
                                   parentTransform, time);
        
        // Datasets 
        std::vector<float> positions;
        std::vector<int> triangles;

        // Copy transformed mesh postion attribute to positions dataset
        positions.resize(numVertices * 3);

        size_t stride_pos_t = bufferView_pos_t.byteStride? bufferView_pos_t.byteStride: 3 * sizeof(float); 

        for (size_t i = 0; i < numVertices; ++i) {
            // Compute offset 
            size_t offset = bufferView_pos_t.byteOffset + accessor_pos_t.byteOffset + i * stride_pos_t;

            // Read vertex position
            memcpy(positions.data() + (i * 3), &buffer_pos_t.data[offset], vertexSize);
        }

        // Triangles dataset
        const tinygltf::Accessor& accessor_triangle = transformedModel.accessors[transformedModel.meshes[0].primitives[0].indices];
        const tinygltf::BufferView& bufferView_triangle = transformedModel.bufferViews[accessor_triangle.bufferView];
        const tinygltf::Buffer& buffer_triangle = transformedModel.buffers[bufferView_triangle.buffer];

        size_t numTriangles = accessor_triangle.count / 3;
        triangles.resize(accessor_triangle.count);

        size_t stride_triangle = bufferView_triangle.byteStride? bufferView_triangle.byteStride: 3 * sizeof(int); 

        for (size_t i = 0; i < numTriangles; ++i) {
            // Compute offset 
            size_t offset = bufferView_triangle.byteOffset + accessor_triangle.byteOffset + i * stride_triangle;

            // Read vertex position
            memcpy(triangles.data() + (i * 3), &buffer_triangle.data[offset], 3 * sizeof(int));
        }

        std::cout << "\tExtracted " << numVertices << " vertices and " << numTriangles << " triangles!\n" << std::endl;

        // Create output directory
        std::string timestepFolder = "output/hdf5/" + formatTimestep(timeStep);
        std::filesystem::create_directories(timestepFolder);

        // Save datasets to HDF5 file
        // File name
        std::string hdf5Filename = timestepFolder + "/" + OBJECT_NAME + "." + formatTimestep(timeStep) + ".p.0.h5";
        
        // Create File
        H5::H5File hdf5File(hdf5Filename, H5F_ACC_TRUNC);

        saveToHDF5(hdf5File, positions, "Position", numVertices, 3);
        saveToHDF5(hdf5File, triangles, "Triangles", numTriangles, 3);

        // Save .xmf file
        std::string xmfFilename = "output/" + OBJECT_NAME + "." + formatTimestep(timeStep) + ".xmf";
        saveToXMF(hdf5Filename, xmfFilename, numVertices, numTriangles);
        
        std::cout << "Successfully extracted mesh at time " << time << " s!\n" << std::endl; 
    }
}


