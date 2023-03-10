#include "perceptron.h"
#include <array>

constexpr int TRAIN_DATA_SZ = 331;
constexpr float X0_TRAIN_MIN = -2.2496;
constexpr float X0_TRAIN_MAX = 2.27832;
constexpr float X1_TRAIN_MIN = -2.36422;
constexpr float X1_TRAIN_MAX = 2.78577;

StaticVec<StaticVec<float, 3>, TRAIN_DATA_SZ> x_train {{
        StaticVec<float, 3>{{1,2}, 2}        
    }, 1};


// FloatMatrix x_train = {{
//     {{-0.212032,-1.566332}, 2},
//     {{-1.645874,1.117465}, 2},
//     {{-1.947735,-1.203657}, 2},
//     {{0.995413,1.770280}, 2},
//     {{0.919947,0.174509}, 2},
//     {{0.919947,0.827325}, 2},
//     {{1.674601,-0.695911}, 2},
//     {{0.467155,-0.768446}, 2},
//     {{0.391690,-0.333236}, 2},
//     {{-0.061102,0.464649}, 2},
//     {{-0.740290,-0.115631}, 2},
//     {{-0.136567,1.044930}, 2},
//     {{0.316225,-0.864918}, 2},
//     {{-1.344012,2.205490}, 2},
//     {{-0.136567,-1.421262}, 2},
//     {{-2.023200,-0.913516}, 2},
//     {{0.618086,-0.043096}, 2},
//     {{-0.061102,-0.260701}, 2},
//     {{0.165294,0.827325}, 2},
//     {{-0.664824,-0.115631}, 2},
//     {{0.693551,1.262535}, 2},
//     {{-0.891220,-0.986051}, 2},
//     {{0.316225,-0.115631}, 2},
//     {{0.316225,1.625210}, 2},
//     {{-0.362963,2.785771}, 2},
//     {{-1.570408,-0.115631}, 2},
//     {{-1.268547,-0.623376}, 2},
//     {{1.448205,-0.115631}, 2},
//     {{0.391690,-0.550841}, 2},
//     {{1.221809,0.537184}, 2},
//     {{0.316225,-1.203657}, 2},
//     {{1.448205,0.899860}, 2},
//     {{0.769017,1.407605}, 2},
//     {{-0.815755,-0.768446}, 2},
//     {{0.769017,0.464649}, 2},
//     {{-1.570408,0.319579}, 2},
//     {{0.693551,0.319579}, 2},
//     {{0.165294,0.319579}, 2},
//     {{-0.287498,-1.469860}, 2},
//     {{-1.042151,-0.986051}, 2},
//     {{0.919947,0.029439}, 2},
//     {{-0.513894,0.899860}, 2},
//     {{-0.136567,-0.043096}, 2},
//     {{-1.721339,-0.405771}, 2},
//     {{1.599135,-0.889580}, 2},
//     {{1.674601,-0.768446}, 2},
//     {{-0.136567,0.247044}, 2},
//     {{-0.966686,-1.566332}, 2},
//     {{-1.117616,-1.566332}, 2},
//     {{-0.815755,0.754790}, 2},
//     {{-0.513894,0.464649}, 2},
//     {{0.467155,1.044930}, 2},
//     {{0.165294,0.464649}, 2},
//     {{0.769017,-1.566332}, 2},
//     {{0.693551,0.609720}, 2},
//     {{0.316225,1.238598}, 2},
//     {{0.844482,0.609720}, 2},
//     {{0.014363,-2.122676}, 2},
//     {{-0.362963,-0.695911}, 2},
//     {{0.316225,-0.043096}, 2},
//     {{1.523670,0.972395}, 2},
//     {{-1.117616,1.842815}, 2},
//     {{-0.136567,0.754790}, 2},
//     {{-1.042151,-0.913516}, 2},
//     {{1.372739,-0.115631}, 2},
//     {{-1.268547,-0.405771}, 2},
//     {{0.240759,-0.986051}, 2},
//     {{-0.589359,-0.623376}, 2},
//     {{-1.419478,0.682255}, 2},
//     {{1.070878,-0.647313}, 2},
//     {{0.542621,0.682255}, 2},
//     {{-0.589359,-1.058586}, 2},
//     {{0.165294,0.609720}, 2},
//     {{-1.117616,0.247044}, 2},
//     {{0.844482,1.335070}, 2},
//     {{-0.362963,0.319579}, 2},
//     {{0.995413,1.407605}, 2},
//     {{1.146343,-1.445198}, 2},
//     {{2.278323,-0.478306}, 2},
//     {{1.448205,0.464649}, 2},
//     {{0.844482,-0.840981}, 2},
//     {{-0.212032,-0.840981}, 2},
//     {{-0.589359,0.537184}, 2},
//     {{-0.589359,-0.840981}, 2},
//     {{-0.438428,-0.550841}, 2},
//     {{-0.891220,1.335070}, 2},
//     {{-0.966686,-1.711402}, 2},
//     {{0.919947,0.658318}, 2},
//     {{-1.117616,-0.550841}, 2},
//     {{-1.947735,-0.333236}, 2},
//     {{-0.815755,-0.986051}, 2},
//     {{-0.589359,-0.526905}, 2},
//     {{-0.061102,1.117465}, 2},
//     {{0.693551,-0.695911}, 2},
//     {{0.089829,-0.478306}, 2},
//     {{-1.645874,-1.203657}, 2},
//     {{-2.023200,-1.421262}, 2},
//     {{1.674601,0.174509}, 2},
//     {{0.089829,0.682255}, 2},
//     {{1.372739,-1.783937}, 2},
//     {{0.240759,-0.333236}, 2},
//     {{-0.966686,1.480140}, 2},
//     {{0.995413,-1.131122}, 2},
//     {{-0.136567,-0.913516}, 2},
//     {{-1.494943,-1.711402}, 2},
//     {{-1.645874,-1.711402}, 2},
//     {{0.618086,-0.333236}, 2},
//     {{0.165294,0.609720}, 2},
//     {{-1.570408,-0.115631}, 2},
//     {{0.844482,-0.478306}, 2},
//     {{-0.061102,-0.115631}, 2},
//     {{0.919947,0.247044}, 2},
//     {{0.089829,-0.913516}, 2},
//     {{1.674601,-0.091694}, 2},
//     {{0.089829,-0.188166}, 2},
//     {{0.693551,0.101974}, 2},
//     {{-0.513894,-0.840981}, 2},
//     {{-0.212032,-1.058586}, 2},
//     {{-2.249596,-0.840981}, 2},
//     {{0.316225,0.754790}, 2},
//     {{0.769017,-0.840981}, 2},
//     {{1.372739,1.238598}, 2},
//     {{1.297274,2.278026}, 2},
//     {{0.240759,-0.840981}, 2},
//     {{-1.042151,0.005503}, 2},
//     {{0.089829,2.060420}, 2},
//     {{0.618086,0.101974}, 2},
//     {{-0.136567,1.842815}, 2},
//     {{0.089829,0.174509}, 2},
//     {{0.844482,0.827325}, 2},
//     {{-0.815755,-1.276192}, 2},
//     {{-2.174131,-0.550841}, 2},
//     {{0.844482,0.174509}, 2},
//     {{-0.362963,1.480140}, 2},
//     {{-0.891220,-0.478306}, 2},
//     {{-0.061102,2.060420}, 2},
//     {{0.240759,-0.043096}, 2},
//     {{0.844482,1.117465}, 2},
//     {{0.391690,0.392114}, 2},
//     {{0.467155,-0.115631}, 2},
//     {{-1.796804,-0.478306}, 2},
//     {{-0.212032,-0.550841}, 2},
//     {{-0.061102,-0.381835}, 2},
//     {{0.165294,-0.550841}, 2},
//     {{-0.212032,-0.840981}, 2},
//     {{0.844482,-1.203657}, 2},
//     {{-0.362963,-0.550841}, 2},
//     {{0.316225,-0.115631}, 2},
//     {{0.316225,0.174509}, 2},
//     {{1.221809,0.609720}, 2},
//     {{-0.061102,-1.566332}, 2},
//     {{0.844482,-0.357173}, 2},
//     {{0.391690,1.335070}, 2},
//     {{0.165294,-1.131122}, 2},
//     {{-0.513894,-1.348727}, 2},
//     {{-0.589359,1.044930}, 2},
//     {{0.542621,0.754790}, 2},
//     {{-1.117616,-0.115631}, 2},
//     {{0.618086,1.044930}, 2},
//     {{1.372739,1.335070}, 2},
//     {{1.825531,0.537184}, 2},
//     {{-1.570408,-0.986051}, 2},
//     {{-1.570408,0.174509}, 2},
//     {{0.014363,-0.478306}, 2},
//     {{0.165294,-0.550841}, 2},
//     {{0.844482,-0.550841}, 2},
//     {{0.391690,-1.203657}, 2},
//     {{1.448205,1.190000}, 2},
//     {{-0.438428,-2.074077}, 2},
//     {{0.919947,1.335070}, 2},
//     {{0.467155,-0.115631}, 2},
//     {{-1.117616,-1.421262}, 2},
//     {{0.542621,2.060420}, 2},
//     {{0.844482,0.730853}, 2},
//     {{-0.589359,-0.840981}, 2},
//     {{-0.513894,-1.711402}, 2},
//     {{0.995413,2.205490}, 2},
//     {{-0.966686,1.262535}, 2},
//     {{-1.193082,-1.058586}, 2},
//     {{1.976462,1.673808}, 2},
//     {{0.693551,1.625210}, 2},
//     {{-0.438428,2.060420}, 2},
//     {{0.693551,-0.260701}, 2},
//     {{1.070878,-1.566332}, 2},
//     {{-1.796804,0.029439}, 2},
//     {{2.278323,0.609720}, 2},
//     {{0.467155,0.537184}, 2},
//     {{-1.419478,-0.550841}, 2},
//     {{0.240759,-0.623376}, 2},
//     {{1.372739,0.101974}, 2},
//     {{-0.891220,-0.768446}, 2},
//     {{0.542621,-1.107185}, 2},
//     {{-0.740290,-0.333236}, 2},
//     {{-1.796804,-0.550841}, 2},
//     {{0.995413,1.480140}, 2},
//     {{0.844482,0.899860}, 2},
//     {{0.618086,0.247044}, 2},
//     {{-0.212032,1.480140}, 2},
//     {{0.014363,-0.115631}, 2},
//     {{0.014363,-1.348727}, 2},
//     {{0.089829,0.464649}, 2},
//     {{-1.117616,1.190000}, 2},
//     {{-1.268547,-0.478306}, 2},
//     {{-0.664824,0.029439}, 2},
//     {{0.391690,-1.493797}, 2},
//     {{-1.494943,-1.566332}, 2},
//     {{-2.174131,-0.478306}, 2},
//     {{0.014363,-0.695911}, 2},
//     {{0.995413,1.842815}, 2},
//     {{0.316225,0.247044}, 2},
//     {{0.844482,1.262535}, 2},
//     {{0.844482,-1.300128}, 2},
//     {{1.523670,0.609720}, 2},
//     {{-0.891220,-0.840981}, 2},
//     {{-1.117616,-1.276192}, 2},
//     {{-0.136567,0.174509}, 2},
//     {{0.391690,0.633656}, 2},
//     {{-0.362963,0.029439}, 2},
//     {{1.523670,2.132956}, 2},
//     {{0.391690,1.166063}, 2},
//     {{1.146343,1.044930}, 2},
//     {{0.769017,0.053376}, 2},
//     {{1.146343,0.174509}, 2},
//     {{1.523670,1.335070}, 2},
//     {{-1.344012,-0.478306}, 2},
//     {{-1.947735,-1.203657}, 2},
//     {{-0.513894,-0.115631}, 2},
//     {{-1.872269,-0.550841}, 2},
//     {{-0.589359,2.278026}, 2},
//     {{-2.249596,-1.421262}, 2},
//     {{0.769017,0.899860}, 2},
//     {{-1.344012,-0.695911}, 2},
//     {{1.372739,1.213936}, 2},
//     {{0.618086,1.625210}, 2},
//     {{0.316225,-0.188166}, 2},
//     {{-1.268547,-0.405771}, 2},
//     {{1.372739,-1.058586}, 2},
//     {{1.221809,-1.783937}, 2},
//     {{-0.891220,-1.131122}, 2},
//     {{1.976462,-1.203657}, 2},
//     {{-1.193082,-0.695911}, 2},
//     {{1.297274,0.464649}, 2},
//     {{-1.117616,-0.840981}, 2},
//     {{1.221809,1.117465}, 2},
//     {{0.014363,-0.260701}, 2},
//     {{0.391690,1.480140}, 2},
//     {{0.089829,1.770280}, 2},
//     {{-0.061102,0.247044}, 2},
//     {{0.995413,-0.526905}, 2},
//     {{0.618086,1.625210}, 2},
//     {{-0.212032,-0.043096}, 2},
//     {{0.919947,-0.695911}, 2},
//     {{-0.664824,0.319579}, 2},
//     {{0.769017,-0.333236}, 2},
//     {{-1.494943,-0.695911}, 2},
//     {{-2.249596,-0.550841}, 2},
//     {{-0.589359,-0.623376}, 2},
//     {{0.618086,0.923796}, 2},
//     {{-0.212032,-1.203657}, 2},
//     {{1.146343,1.190000}, 2},
//     {{0.919947,1.190000}, 2},
//     {{-1.117616,-0.550841}, 2},
//     {{0.467155,-0.986051}, 2},
//     {{0.089829,-1.179720}, 2},
//     {{-1.645874,-0.260701}, 2},
//     {{-1.344012,-0.115631}, 2},
//     {{-0.513894,1.915350}, 2},
//     {{0.165294,-0.115631}, 2},
//     {{0.995413,0.464649}, 2},
//     {{0.316225,-1.203657}, 2},
//     {{-1.570408,-0.695911}, 2},
//     {{-2.023200,0.174509}, 2},
//     {{-2.023200,-1.566332}, 2},
//     {{-1.570408,-0.913516}, 2},
//     {{0.919947,-1.131122}, 2},
//     {{0.165294,0.899860}, 2},
//     {{1.221809,1.842815}, 2},
//     {{1.372739,0.754790}, 2},
//     {{-1.872269,-0.768446}, 2},
//     {{0.467155,1.117465}, 2},
//     {{1.146343,-0.164229}, 2},
//     {{1.297274,1.480140}, 2},
//     {{0.919947,0.464649}, 2},
//     {{-0.212032,-0.695911}, 2},
//     {{-0.438428,1.915350}, 2},
//     {{0.316225,0.464649}, 2},
//     {{0.391690,1.480140}, 2},
//     {{0.467155,0.005503}, 2},
//     {{-0.664824,1.480140}, 2},
//     {{-0.966686,0.029439}, 2},
//     {{0.240759,-0.236764}, 2},
//     {{0.240759,0.899860}, 2},
//     {{0.618086,1.190000}, 2},
//     {{-1.193082,0.537184}, 2},
//     {{0.089829,-0.019159}, 2},
//     {{0.240759,1.044930}, 2},
//     {{0.542621,-1.155058}, 2},
//     {{0.240759,-0.695911}, 2},
//     {{-1.872269,-0.405771}, 2},
//     {{1.674601,0.754790}, 2},
//     {{0.769017,0.101974}, 2},
//     {{1.900997,0.464649}, 2},
//     {{-0.438428,-1.131122}, 2},
//     {{1.448205,0.803388}, 2},
//     {{0.919947,-0.333236}, 2},
//     {{0.316225,-0.478306}, 2},
//     {{-1.268547,-0.309299}, 2},
//     {{0.089829,-0.260701}, 2},
//     {{0.316225,-0.913516}, 2},
//     {{0.014363,-0.405771}, 2},
//     {{-1.117616,1.697745}, 2},
//     {{-0.061102,0.029439}, 2},
//     {{-0.589359,-2.364217}, 2},
//     {{0.014363,-0.405771}, 2},
//     {{-1.042151,-0.986051}, 2},
//     {{1.372739,-0.550841}, 2},
//     {{0.014363,-0.115631}, 2},
//     {{-0.212032,0.174509}, 2},
//     {{0.316225,1.335070}, 2},
//     {{-1.494943,-0.840981}, 2},
//     {{-0.061102,-1.397325}, 2},
//     {{-0.589359,-0.623376}, 2},
//     {{-0.287498,1.117465}, 2},
//     {{-0.664824,-0.188166}, 2},
//     {{-1.117616,-0.768446}, 2},
//     {{-1.042151,0.247044}, 2},
//     {{0.316225,0.029439}, 2},
//     {{1.750066,-0.115631}, 2},
//     {{0.844482,0.368178}, 2},
//     {{0.542621,0.609720}, 2},
//     {{-0.891220,-0.115631}, 2}}, TRAIN_DATA_SZ
// };
//
float y_train[TRAIN_DATA_SZ] = {
    0.087227,
    0.647975,
    0.186916,
    0.797508,
    0.728972,
    0.660436,
    0.355140,
    0.068536,
    0.146417,
    0.046729,
    0.130841,
    0.439252,
    0.778816,
    0.775701,
    0.246106,
    0.339564,
    0.193146,
    0.071651,
    0.747663,
    0.635514,
    0.722741,
    0.105919,
    0.685358,
    0.330218,
    0.595016,
    0.370717,
    0.725857,
    0.638629,
    0.105919,
    0.956386,
    0.124611,
    0.367601,
    0.161994,
    0.224299,
    0.582554,
    0.109034,
    0.404984,
    0.517134,
    0.221184,
    1.000000,
    0.361371,
    0.707165,
    0.423676,
    0.348910,
    0.199377,
    0.542056,
    0.214953,
    0.548287,
    0.457944,
    0.255452,
    0.769470,
    0.604361,
    0.401869,
    0.196262,
    0.984424,
    0.084112,
    0.489097,
    0.124611,
    0.485981,
    0.545171,
    0.672897,
    0.679128,
    0.261682,
    0.133956,
    0.535825,
    0.221184,
    0.161994,
    0.087227,
    0.190031,
    0.538941,
    0.087227,
    0.037383,
    0.831776,
    0.062305,
    0.143302,
    0.305296,
    0.778816,
    0.146417,
    0.445483,
    0.171340,
    0.264798,
    0.442368,
    0.118380,
    0.445483,
    0.202492,
    0.728972,
    0.099688,
    0.685358,
    0.280374,
    0.862928,
    0.576324,
    0.791277,
    0.613707,
    0.373832,
    0.707165,
    0.364486,
    0.414330,
    0.389408,
    0.355140,
    0.239875,
    0.647975,
    0.598131,
    0.121495,
    0.476635,
    0.426791,
    0.395639,
    0.464174,
    0.551402,
    0.445483,
    0.246106,
    0.299065,
    0.339564,
    0.345794,
    0.330218,
    0.196262,
    0.137072,
    0.186916,
    0.492212,
    0.545171,
    0.127726,
    0.420561,
    0.308411,
    0.510903,
    0.227414,
    0.102804,
    0.794392,
    0.473520,
    0.529595,
    0.498442,
    0.654206,
    0.165109,
    0.161994,
    0.841121,
    0.834891,
    0.214953,
    0.707165,
    0.267913,
    0.710280,
    0.654206,
    0.165109,
    0.056075,
    0.052960,
    0.183801,
    0.183801,
    0.308411,
    0.043614,
    0.084112,
    0.476635,
    0.074766,
    0.623053,
    0.202492,
    0.224299,
    0.847352,
    0.224299,
    0.118380,
    0.501558,
    0.130841,
    0.105919,
    0.570093,
    0.803738,
    0.261682,
    0.283489,
    0.236760,
    0.074766,
    0.193146,
    0.174455,
    0.118380,
    0.370717,
    0.152648,
    0.498442,
    0.093458,
    0.052960,
    0.741433,
    0.202492,
    0.084112,
    0.105919,
    0.242991,
    0.429906,
    0.071651,
    0.638629,
    0.968847,
    0.071651,
    0.261682,
    0.330218,
    0.074766,
    0.785047,
    0.819315,
    0.367601,
    0.080997,
    0.457944,
    0.320872,
    0.358255,
    0.043614,
    0.688473,
    0.778816,
    0.289720,
    0.697819,
    0.246106,
    0.196262,
    0.018692,
    0.342679,
    0.485981,
    0.146417,
    0.476635,
    0.208723,
    0.246106,
    0.143302,
    0.277259,
    0.887850,
    0.068536,
    0.476635,
    0.165109,
    0.660436,
    0.180685,
    0.420561,
    0.299065,
    0.644860,
    0.000000,
    0.844237,
    0.514019,
    0.389408,
    0.174455,
    0.205607,
    0.694704,
    0.495327,
    0.214953,
    0.302181,
    0.230530,
    0.679128,
    0.358255,
    0.317757,
    0.588785,
    0.258567,
    0.404984,
    0.255452,
    0.504673,
    0.202492,
    0.280374,
    0.370717,
    0.482866,
    0.514019,
    0.461059,
    0.186916,
    0.429906,
    0.208723,
    0.205607,
    0.551402,
    0.180685,
    0.623053,
    0.887850,
    0.364486,
    0.330218,
    0.230530,
    0.180685,
    0.887850,
    0.348910,
    0.311526,
    0.295950,
    0.781931,
    0.476635,
    0.352025,
    0.364486,
    0.323988,
    0.205607,
    0.202492,
    0.236760,
    0.467290,
    0.461059,
    0.482866,
    0.211838,
    0.242991,
    0.236760,
    0.545171,
    0.137072,
    0.433022,
    0.682243,
    0.797508,
    0.149533,
    0.563863,
    0.797508,
    0.414330,
    0.744548,
    0.588785,
    0.389408,
    0.274143,
    0.797508,
    0.922118,
    0.376947,
    0.451713,
    0.607477,
    0.728972,
    0.077882,
    0.741433,
    0.479751,
    0.183801,
    0.694704,
    0.370717,
    0.676012,
    0.498442,
    0.532710,
    0.545171,
    0.140187,
    0.052960,
    0.607477,
    0.604361,
    0.747663,
    0.112150,
    0.221184,
    0.392523,
    0.267913,
    0.454829,
    0.168224,
    0.161994,
    0.738318,
    0.314642,
    0.317757,
    0.398754,
    0.573209,
    0.239875,
    0.112150,
    0.520249,
    0.146417,
    0.591900,
    0.109034,
    0.261682,
    0.545171,
    0.607477,
    0.361371,
    0.333333,
    0.433022,
    0.364486 
};

