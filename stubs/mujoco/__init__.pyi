"""
Python bindings for MuJoCo.
"""
from __future__ import annotations

import ctypes as ctypes
import os as os
import platform as platform
import subprocess as subprocess
import typing
import warnings as warnings
import zipfile as zipfile
from typing import IO, Any

from mujoco._callbacks import (
    get_mjcb_act_bias,
    get_mjcb_act_dyn,
    get_mjcb_act_gain,
    get_mjcb_contactfilter,
    get_mjcb_control,
    get_mjcb_passive,
    get_mjcb_sensor,
    get_mjcb_time,
    get_mju_user_free,
    get_mju_user_malloc,
    get_mju_user_warning,
    set_mjcb_act_bias,
    set_mjcb_act_dyn,
    set_mjcb_act_gain,
    set_mjcb_contactfilter,
    set_mjcb_control,
    set_mjcb_passive,
    set_mjcb_sensor,
    set_mjcb_time,
    set_mju_user_free,
    set_mju_user_malloc,
    set_mju_user_warning,
)
from mujoco._enums import (
    mjtAlignFree,
    mjtBias,
    mjtBuiltin,
    mjtButton,
    mjtCamera,
    mjtCamLight,
    mjtCatBit,
    mjtColorSpace,
    mjtConDataField,
    mjtCone,
    mjtConstraint,
    mjtConstraintState,
    mjtDataType,
    mjtDepthMap,
    mjtDisableBit,
    mjtDyn,
    mjtEnableBit,
    mjtEq,
    mjtEvent,
    mjtFlexSelf,
    mjtFont,
    mjtFontScale,
    mjtFrame,
    mjtFramebuffer,
    mjtGain,
    mjtGeom,
    mjtGeomInertia,
    mjtGridPos,
    mjtInertiaFromGeom,
    mjtIntegrator,
    mjtItem,
    mjtJacobian,
    mjtJoint,
    mjtLabel,
    mjtLightType,
    mjtLimited,
    mjtLRMode,
    mjtMark,
    mjtMeshBuiltin,
    mjtMeshInertia,
    mjtMouse,
    mjtObj,
    mjtOrientation,
    mjtPertBit,
    mjtPluginCapabilityBit,
    mjtRndFlag,
    mjtSameFrame,
    mjtSDFType,
    mjtSection,
    mjtSensor,
    mjtSolver,
    mjtStage,
    mjtState,
    mjtStereo,
    mjtTaskStatus,
    mjtTexture,
    mjtTextureRole,
    mjtTimer,
    mjtTrn,
    mjtVisFlag,
    mjtWarning,
    mjtWrap,
)
from mujoco._functions import (
    mj_addContact,
    mj_addM,
    mj_angmomMat,
    mj_applyFT,
    mj_camlight,
    mj_checkAcc,
    mj_checkPos,
    mj_checkVel,
    mj_clearCache,
    mj_collision,
    mj_compareFwdInv,
    mj_comPos,
    mj_comVel,
    mj_constraintUpdate,
    mj_contactForce,
    mj_crb,
    mj_defaultLROpt,
    mj_defaultOption,
    mj_defaultSolRefImp,
    mj_defaultVisual,
    mj_differentiatePos,
    mj_energyPos,
    mj_energyVel,
    mj_Euler,
    mj_factorM,
    mj_flex,
    mj_forward,
    mj_forwardSkip,
    mj_fullM,
    mj_fwdAcceleration,
    mj_fwdActuation,
    mj_fwdConstraint,
    mj_fwdPosition,
    mj_fwdVelocity,
    mj_geomDistance,
    mj_getCache,
    mj_getCacheCapacity,
    mj_getCacheSize,
    mj_getState,
    mj_getTotalmass,
    mj_id2name,
    mj_implicit,
    mj_integratePos,
    mj_invConstraint,
    mj_inverse,
    mj_inverseSkip,
    mj_invPosition,
    mj_invVelocity,
    mj_isDual,
    mj_island,
    mj_isPyramidal,
    mj_isSparse,
    mj_jac,
    mj_jacBody,
    mj_jacBodyCom,
    mj_jacDot,
    mj_jacGeom,
    mj_jacPointAxis,
    mj_jacSite,
    mj_jacSubtreeCom,
    mj_kinematics,
    mj_loadAllPluginLibraries,
    mj_loadPluginLibrary,
    mj_local2Global,
    mj_makeConstraint,
    mj_makeM,
    mj_mulJacTVec,
    mj_mulJacVec,
    mj_mulM,
    mj_mulM2,
    mj_multiRay,
    mj_name2id,
    mj_normalizeQuat,
    mj_objectAcceleration,
    mj_objectVelocity,
    mj_passive,
    mj_printData,
    mj_printFormattedData,
    mj_printFormattedModel,
    mj_printFormattedScene,
    mj_printModel,
    mj_printScene,
    mj_printSchema,
    mj_projectConstraint,
    mj_ray,
    mj_rayHfield,
    mj_rayMesh,
    mj_referenceConstraint,
    mj_resetCallbacks,
    mj_resetData,
    mj_resetDataDebug,
    mj_resetDataKeyframe,
    mj_rne,
    mj_rnePostConstraint,
    mj_RungeKutta,
    mj_saveLastXML,
    mj_saveModel,
    mj_sensorAcc,
    mj_sensorPos,
    mj_sensorVel,
    mj_setCacheCapacity,
    mj_setConst,
    mj_setKeyframe,
    mj_setLengthRange,
    mj_setState,
    mj_setTotalmass,
    mj_sizeModel,
    mj_solveM,
    mj_solveM2,
    mj_stateSize,
    mj_step,
    mj_step1,
    mj_step2,
    mj_subtreeVel,
    mj_tendon,
    mj_transmission,
    mj_version,
    mj_versionString,
    mjd_inverseFD,
    mjd_quatIntegrate,
    mjd_subQuat,
    mjd_transitionFD,
    mju_add,
    mju_add3,
    mju_addScl,
    mju_addScl3,
    mju_addTo,
    mju_addTo3,
    mju_addToScl,
    mju_addToScl3,
    mju_axisAngle2Quat,
    mju_band2Dense,
    mju_bandDiag,
    mju_bandMulMatVec,
    mju_boxQP,
    mju_cholFactor,
    mju_cholFactorBand,
    mju_cholSolve,
    mju_cholSolveBand,
    mju_cholUpdate,
    mju_clip,
    mju_copy,
    mju_copy3,
    mju_copy4,
    mju_cross,
    mju_d2n,
    mju_decodePyramid,
    mju_dense2Band,
    mju_dense2sparse,
    mju_derivQuat,
    mju_dist3,
    mju_dot,
    mju_dot3,
    mju_eig3,
    mju_encodePyramid,
    mju_euler2Quat,
    mju_eye,
    mju_f2n,
    mju_fill,
    mju_getXMLDependencies,
    mju_Halton,
    mju_insertionSort,
    mju_insertionSortInt,
    mju_isBad,
    mju_isZero,
    mju_L1,
    mju_mat2Quat,
    mju_mat2Rot,
    mju_max,
    mju_min,
    mju_mulMatMat,
    mju_mulMatMatT,
    mju_mulMatTMat,
    mju_mulMatTVec,
    mju_mulMatTVec3,
    mju_mulMatVec,
    mju_mulMatVec3,
    mju_mulPose,
    mju_mulQuat,
    mju_mulQuatAxis,
    mju_mulVecMatVec,
    mju_muscleBias,
    mju_muscleDynamics,
    mju_muscleGain,
    mju_n2d,
    mju_n2f,
    mju_negPose,
    mju_negQuat,
    mju_norm,
    mju_norm3,
    mju_normalize,
    mju_normalize3,
    mju_normalize4,
    mju_printMat,
    mju_printMatSparse,
    mju_quat2Mat,
    mju_quat2Vel,
    mju_quatIntegrate,
    mju_quatZ2Vec,
    mju_rayFlex,
    mju_rayGeom,
    mju_raySkin,
    mju_rotVecQuat,
    mju_round,
    mju_scl,
    mju_scl3,
    mju_sigmoid,
    mju_sign,
    mju_sparse2dense,
    mju_springDamper,
    mju_sqrMatTD,
    mju_standardNormal,
    mju_str2Type,
    mju_sub,
    mju_sub3,
    mju_subFrom,
    mju_subFrom3,
    mju_subQuat,
    mju_sum,
    mju_symmetrize,
    mju_transformSpatial,
    mju_transpose,
    mju_trnVecPose,
    mju_type2Str,
    mju_unit4,
    mju_warningText,
    mju_writeLog,
    mju_writeNumBytes,
    mju_zero,
    mju_zero3,
    mju_zero4,
    mjv_addGeoms,
    mjv_alignToCamera,
    mjv_applyPerturbForce,
    mjv_applyPerturbPose,
    mjv_cameraInModel,
    mjv_cameraInRoom,
    mjv_connector,
    mjv_defaultCamera,
    mjv_defaultFigure,
    mjv_defaultFreeCamera,
    mjv_defaultOption,
    mjv_defaultPerturb,
    mjv_frustumHeight,
    mjv_initGeom,
    mjv_initPerturb,
    mjv_makeLights,
    mjv_model2room,
    mjv_moveCamera,
    mjv_moveModel,
    mjv_movePerturb,
    mjv_room2model,
    mjv_select,
    mjv_updateCamera,
    mjv_updateScene,
    mjv_updateSkin,
)
from mujoco._render import (
    MjrContext,
    MjrRect,
    mjr_addAux,
    mjr_blitAux,
    mjr_blitBuffer,
    mjr_changeFont,
    mjr_drawPixels,
    mjr_figure,
    mjr_findRect,
    mjr_finish,
    mjr_getError,
    mjr_label,
    mjr_maxViewport,
    mjr_overlay,
    mjr_readPixels,
    mjr_rectangle,
    mjr_render,
    mjr_resizeOffscreen,
    mjr_restoreBuffer,
    mjr_setAux,
    mjr_setBuffer,
    mjr_text,
    mjr_uploadHField,
    mjr_uploadMesh,
    mjr_uploadTexture,
)
from mujoco._specs import (
    MjByteVec,
    MjCharVec,
    MjDoubleVec,
    MjFloatVec,
    MjIntVec,
    MjsActuator,
    MjsBody,
    MjsCamera,
    MjsCompiler,
    MjsDefault,
    MjsElement,
    MjsEquality,
    MjsExclude,
    MjsFlex,
    MjsFrame,
    MjsGeom,
    MjsHField,
    MjsJoint,
    MjsKey,
    MjsLight,
    MjsMaterial,
    MjsMesh,
    MjsNumeric,
    MjsOrientation,
    MjsPair,
    MjSpec,
    MjsPlugin,
    MjsSensor,
    MjsSite,
    MjsSkin,
    MjsTendon,
    MjsText,
    MjsTexture,
    MjStringVec,
    MjsTuple,
    MjsWrap,
    MjVisualHeadlight,
    MjVisualRgba,
)
from mujoco._structs import (
    MjContact,
    MjData,
    MjLROpt,
    MjModel,
    MjOption,
    MjSolverStat,
    MjStatistic,
    MjTimerStat,
    MjvCamera,
    MjvFigure,
    MjvGeom,
    MjvGLCamera,
    MjVisual,
    MjvLight,
    MjvOption,
    MjvPerturb,
    MjvScene,
    MjWarningStat,
    mjv_averageCamera,
)
from mujoco.glfw import GLContext
from mujoco.renderer import Renderer

from . import (
    _specs,
    _structs,
    gl_context,
    glfw,
    renderer,
)

__all__: list[str] = ['Any', 'FatalError', 'GLContext', 'HEADERS_DIR', 'IO', 'MjByteVec', 'MjCharVec', 'MjContact', 'MjData', 'MjDoubleVec', 'MjFloatVec', 'MjIntVec', 'MjLROpt', 'MjModel', 'MjOption', 'MjSolverStat', 'MjSpec', 'MjStatistic', 'MjStringVec', 'MjStruct', 'MjTimerStat', 'MjVisual', 'MjVisualHeadlight', 'MjVisualRgba', 'MjWarningStat', 'MjrContext', 'MjrRect', 'MjsActuator', 'MjsBody', 'MjsCamera', 'MjsCompiler', 'MjsDefault', 'MjsElement', 'MjsEquality', 'MjsExclude', 'MjsFlex', 'MjsFrame', 'MjsGeom', 'MjsHField', 'MjsJoint', 'MjsKey', 'MjsLight', 'MjsMaterial', 'MjsMesh', 'MjsNumeric', 'MjsOrientation', 'MjsPair', 'MjsPlugin', 'MjsSensor', 'MjsSite', 'MjsSkin', 'MjsTendon', 'MjsText', 'MjsTexture', 'MjsTuple', 'MjsWrap', 'MjvCamera', 'MjvFigure', 'MjvGLCamera', 'MjvGeom', 'MjvLight', 'MjvOption', 'MjvPerturb', 'MjvScene', 'PLUGINS_DIR', 'PLUGIN_HANDLES', 'Renderer', 'UnexpectedError', 'ctypes', 'from_zip', 'get_mjcb_act_bias', 'get_mjcb_act_dyn', 'get_mjcb_act_gain', 'get_mjcb_contactfilter', 'get_mjcb_control', 'get_mjcb_passive', 'get_mjcb_sensor', 'get_mjcb_time', 'get_mju_user_free', 'get_mju_user_malloc', 'get_mju_user_warning', 'gl_context', 'glfw', 'mjDISABLESTRING', 'mjENABLESTRING', 'mjFRAMESTRING', 'mjLABELSTRING', 'mjMAXCONPAIR', 'mjMAXFLEXNODES', 'mjMAXIMP', 'mjMAXLIGHT', 'mjMAXLINE', 'mjMAXLINEPNT', 'mjMAXOVERLAY', 'mjMAXPLANEGRID', 'mjMAXTREEDEPTH', 'mjMAXVAL', 'mjMINIMP', 'mjMINMU', 'mjMINVAL', 'mjNBIAS', 'mjNDYN', 'mjNEQDATA', 'mjNFLUID', 'mjNGAIN', 'mjNGROUP', 'mjNIMP', 'mjNISLAND', 'mjNREF', 'mjNSENS', 'mjNSOLVER', 'mjPI', 'mjRNDSTRING', 'mjTIMERSTRING', 'mjVERSION_HEADER', 'mjVISSTRING', 'mj_Euler', 'mj_RungeKutta', 'mj_addContact', 'mj_addM', 'mj_angmomMat', 'mj_applyFT', 'mj_camlight', 'mj_checkAcc', 'mj_checkPos', 'mj_checkVel', 'mj_clearCache', 'mj_collision', 'mj_comPos', 'mj_comVel', 'mj_compareFwdInv', 'mj_constraintUpdate', 'mj_contactForce', 'mj_crb', 'mj_defaultLROpt', 'mj_defaultOption', 'mj_defaultSolRefImp', 'mj_defaultVisual', 'mj_differentiatePos', 'mj_energyPos', 'mj_energyVel', 'mj_factorM', 'mj_flex', 'mj_forward', 'mj_forwardSkip', 'mj_fullM', 'mj_fwdAcceleration', 'mj_fwdActuation', 'mj_fwdConstraint', 'mj_fwdPosition', 'mj_fwdVelocity', 'mj_geomDistance', 'mj_getCache', 'mj_getCacheCapacity', 'mj_getCacheSize', 'mj_getState', 'mj_getTotalmass', 'mj_id2name', 'mj_implicit', 'mj_integratePos', 'mj_invConstraint', 'mj_invPosition', 'mj_invVelocity', 'mj_inverse', 'mj_inverseSkip', 'mj_isDual', 'mj_isPyramidal', 'mj_isSparse', 'mj_island', 'mj_jac', 'mj_jacBody', 'mj_jacBodyCom', 'mj_jacDot', 'mj_jacGeom', 'mj_jacPointAxis', 'mj_jacSite', 'mj_jacSubtreeCom', 'mj_kinematics', 'mj_loadAllPluginLibraries', 'mj_loadPluginLibrary', 'mj_local2Global', 'mj_makeConstraint', 'mj_makeM', 'mj_mulJacTVec', 'mj_mulJacVec', 'mj_mulM', 'mj_mulM2', 'mj_multiRay', 'mj_name2id', 'mj_normalizeQuat', 'mj_objectAcceleration', 'mj_objectVelocity', 'mj_passive', 'mj_printData', 'mj_printFormattedData', 'mj_printFormattedModel', 'mj_printFormattedScene', 'mj_printModel', 'mj_printScene', 'mj_printSchema', 'mj_projectConstraint', 'mj_ray', 'mj_rayHfield', 'mj_rayMesh', 'mj_referenceConstraint', 'mj_resetCallbacks', 'mj_resetData', 'mj_resetDataDebug', 'mj_resetDataKeyframe', 'mj_rne', 'mj_rnePostConstraint', 'mj_saveLastXML', 'mj_saveModel', 'mj_sensorAcc', 'mj_sensorPos', 'mj_sensorVel', 'mj_setCacheCapacity', 'mj_setConst', 'mj_setKeyframe', 'mj_setLengthRange', 'mj_setState', 'mj_setTotalmass', 'mj_sizeModel', 'mj_solveM', 'mj_solveM2', 'mj_stateSize', 'mj_step', 'mj_step1', 'mj_step2', 'mj_subtreeVel', 'mj_tendon', 'mj_transmission', 'mj_version', 'mj_versionString', 'mjd_inverseFD', 'mjd_quatIntegrate', 'mjd_subQuat', 'mjd_transitionFD', 'mjr_addAux', 'mjr_blitAux', 'mjr_blitBuffer', 'mjr_changeFont', 'mjr_drawPixels', 'mjr_figure', 'mjr_findRect', 'mjr_finish', 'mjr_getError', 'mjr_label', 'mjr_maxViewport', 'mjr_overlay', 'mjr_readPixels', 'mjr_rectangle', 'mjr_render', 'mjr_resizeOffscreen', 'mjr_restoreBuffer', 'mjr_setAux', 'mjr_setBuffer', 'mjr_text', 'mjr_uploadHField', 'mjr_uploadMesh', 'mjr_uploadTexture', 'mjtAlignFree', 'mjtBias', 'mjtBuiltin', 'mjtButton', 'mjtCamLight', 'mjtCamera', 'mjtCatBit', 'mjtColorSpace', 'mjtConDataField', 'mjtCone', 'mjtConstraint', 'mjtConstraintState', 'mjtDataType', 'mjtDepthMap', 'mjtDisableBit', 'mjtDyn', 'mjtEnableBit', 'mjtEq', 'mjtEvent', 'mjtFlexSelf', 'mjtFont', 'mjtFontScale', 'mjtFrame', 'mjtFramebuffer', 'mjtGain', 'mjtGeom', 'mjtGeomInertia', 'mjtGridPos', 'mjtInertiaFromGeom', 'mjtIntegrator', 'mjtItem', 'mjtJacobian', 'mjtJoint', 'mjtLRMode', 'mjtLabel', 'mjtLightType', 'mjtLimited', 'mjtMark', 'mjtMeshBuiltin', 'mjtMeshInertia', 'mjtMouse', 'mjtObj', 'mjtOrientation', 'mjtPertBit', 'mjtPluginCapabilityBit', 'mjtRndFlag', 'mjtSDFType', 'mjtSameFrame', 'mjtSection', 'mjtSensor', 'mjtSolver', 'mjtStage', 'mjtState', 'mjtStereo', 'mjtTaskStatus', 'mjtTexture', 'mjtTextureRole', 'mjtTimer', 'mjtTrn', 'mjtVisFlag', 'mjtWarning', 'mjtWrap', 'mju_Halton', 'mju_L1', 'mju_add', 'mju_add3', 'mju_addScl', 'mju_addScl3', 'mju_addTo', 'mju_addTo3', 'mju_addToScl', 'mju_addToScl3', 'mju_axisAngle2Quat', 'mju_band2Dense', 'mju_bandDiag', 'mju_bandMulMatVec', 'mju_boxQP', 'mju_cholFactor', 'mju_cholFactorBand', 'mju_cholSolve', 'mju_cholSolveBand', 'mju_cholUpdate', 'mju_clip', 'mju_copy', 'mju_copy3', 'mju_copy4', 'mju_cross', 'mju_d2n', 'mju_decodePyramid', 'mju_dense2Band', 'mju_dense2sparse', 'mju_derivQuat', 'mju_dist3', 'mju_dot', 'mju_dot3', 'mju_eig3', 'mju_encodePyramid', 'mju_euler2Quat', 'mju_eye', 'mju_f2n', 'mju_fill', 'mju_getXMLDependencies', 'mju_insertionSort', 'mju_insertionSortInt', 'mju_isBad', 'mju_isZero', 'mju_mat2Quat', 'mju_mat2Rot', 'mju_max', 'mju_min', 'mju_mulMatMat', 'mju_mulMatMatT', 'mju_mulMatTMat', 'mju_mulMatTVec', 'mju_mulMatTVec3', 'mju_mulMatVec', 'mju_mulMatVec3', 'mju_mulPose', 'mju_mulQuat', 'mju_mulQuatAxis', 'mju_mulVecMatVec', 'mju_muscleBias', 'mju_muscleDynamics', 'mju_muscleGain', 'mju_n2d', 'mju_n2f', 'mju_negPose', 'mju_negQuat', 'mju_norm', 'mju_norm3', 'mju_normalize', 'mju_normalize3', 'mju_normalize4', 'mju_printMat', 'mju_printMatSparse', 'mju_quat2Mat', 'mju_quat2Vel', 'mju_quatIntegrate', 'mju_quatZ2Vec', 'mju_rayFlex', 'mju_rayGeom', 'mju_raySkin', 'mju_rotVecQuat', 'mju_round', 'mju_scl', 'mju_scl3', 'mju_sigmoid', 'mju_sign', 'mju_sparse2dense', 'mju_springDamper', 'mju_sqrMatTD', 'mju_standardNormal', 'mju_str2Type', 'mju_sub', 'mju_sub3', 'mju_subFrom', 'mju_subFrom3', 'mju_subQuat', 'mju_sum', 'mju_symmetrize', 'mju_transformSpatial', 'mju_transpose', 'mju_trnVecPose', 'mju_type2Str', 'mju_unit4', 'mju_warningText', 'mju_writeLog', 'mju_writeNumBytes', 'mju_zero', 'mju_zero3', 'mju_zero4', 'mjv_addGeoms', 'mjv_alignToCamera', 'mjv_applyPerturbForce', 'mjv_applyPerturbPose', 'mjv_averageCamera', 'mjv_cameraInModel', 'mjv_cameraInRoom', 'mjv_connector', 'mjv_defaultCamera', 'mjv_defaultFigure', 'mjv_defaultFreeCamera', 'mjv_defaultOption', 'mjv_defaultPerturb', 'mjv_frustumHeight', 'mjv_initGeom', 'mjv_initPerturb', 'mjv_makeLights', 'mjv_model2room', 'mjv_moveCamera', 'mjv_moveModel', 'mjv_movePerturb', 'mjv_room2model', 'mjv_select', 'mjv_updateCamera', 'mjv_updateScene', 'mjv_updateSkin', 'os', 'platform', 'renderer', 'set_mjcb_act_bias', 'set_mjcb_act_dyn', 'set_mjcb_act_gain', 'set_mjcb_contactfilter', 'set_mjcb_control', 'set_mjcb_passive', 'set_mjcb_sensor', 'set_mjcb_time', 'set_mju_user_free', 'set_mju_user_malloc', 'set_mju_user_warning', 'subprocess', 'to_zip', 'warnings', 'zipfile']
class FatalError(Exception):
    pass
class UnexpectedError(Exception):
    pass
class _MjBindData:
    def __getattr__(self, key: str):
        ...
    def __init__(self, elements: typing.Sequence[typing.Any]):
        ...
class _MjBindModel:
    def __getattr__(self, key: str):
        ...
    def __init__(self, elements: typing.Sequence[typing.Any]):
        ...
def _bind_data(data: _structs.MjData, specs: typing.Union[typing.Sequence[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]], mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]):
    """
    Bind a Mujoco spec to a mjData.

    Args:
      data: The mjData to bind to.
      specs: The mjSpec elements to use for binding, can be a single element or a
        sequence.
    Returns:
      A MjDataGroupedViews object or a list of the same type.
    """
def _bind_model(model: _structs.MjModel, specs: typing.Union[typing.Sequence[typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]], mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]):
    """
    Bind a Mujoco spec to a mjModel.

    Args:
      model: The mjModel to bind to.
      specs: The mjSpec elements to use for binding, can be a single element or a
        sequence.
    Returns:
      A MjModelGroupedViews object or a list of the same type.
    """
def _load_all_bundled_plugins():
    ...
def from_zip(file: typing.Union[str, typing.IO[bytes]]) -> _specs.MjSpec:
    """
    Reads a zip file and returns an MjSpec.

    Args:
      file: The path to the file to read from or the file object to read from.
    Returns:
      An MjSpec object.
    """
def to_zip(spec: _specs.MjSpec, file: typing.Union[str, typing.IO[bytes]]) -> None:
    """
    Converts an MjSpec to a zip file.

    Args:
      spec: The mjSpec to save to a file.
      file: The path to the file to save to or the file object to write to.
    """
HEADERS_DIR: str = '/home/ted/Code/lerax/.venv/lib/python3.13/site-packages/mujoco/include/mujoco'
MjStruct: typing._UnionGenericAlias  # value = typing.Union[mujoco._specs.MjsBody, mujoco._specs.MjsFrame, mujoco._specs.MjsGeom, mujoco._specs.MjsJoint, mujoco._specs.MjsLight, mujoco._specs.MjsMaterial, mujoco._specs.MjsSite, mujoco._specs.MjsMesh, mujoco._specs.MjsSkin, mujoco._specs.MjsTexture, mujoco._specs.MjsText, mujoco._specs.MjsTuple, mujoco._specs.MjsCamera, mujoco._specs.MjsFlex, mujoco._specs.MjsHField, mujoco._specs.MjsKey, mujoco._specs.MjsNumeric, mujoco._specs.MjsPair, mujoco._specs.MjsExclude, mujoco._specs.MjsEquality, mujoco._specs.MjsTendon, mujoco._specs.MjsSensor, mujoco._specs.MjsActuator, mujoco._specs.MjsPlugin]
PLUGINS_DIR: str = '/home/ted/Code/lerax/.venv/lib/python3.13/site-packages/mujoco/plugin'
PLUGIN_HANDLES: list  # value = [<CDLL '/home/ted/Code/lerax/.venv/lib/python3.13/site-packages/mujoco/plugin/libactuator.so', handle 5558c9a97e80 at 0x7f4c767a5e50>, <CDLL '/home/ted/Code/lerax/.venv/lib/python3.13/site-packages/mujoco/plugin/libsdf_plugin.so', handle 5558c9a98b50 at 0x7f4c72969810>, <CDLL '/home/ted/Code/lerax/.venv/lib/python3.13/site-packages/mujoco/plugin/libelasticity.so', handle 5558c9aa3640 at 0x7f4c72969950>, <CDLL '/home/ted/Code/lerax/.venv/lib/python3.13/site-packages/mujoco/plugin/libsensor.so', handle 5558c9aa4370 at 0x7f4c72969a90>]
_SYSTEM: str = 'Linux'
__version__: str = '3.3.7'
mjDISABLESTRING: tuple = ('Constraint', 'Equality', 'Frictionloss', 'Limit', 'Contact', 'Spring', 'Damper', 'Gravity', 'Clampctrl', 'Warmstart', 'Filterparent', 'Actuation', 'Refsafe', 'Sensor', 'Midphase', 'Eulerdamp', 'AutoReset', 'NativeCCD', 'Island')
mjENABLESTRING: tuple = ('Override', 'Energy', 'Fwdinv', 'InvDiscrete', 'MultiCCD')
mjFRAMESTRING: tuple = ('None', 'Body', 'Geom', 'Site', 'Camera', 'Light', 'Contact', 'World')
mjLABELSTRING: tuple = ('None', 'Body', 'Joint', 'Geom', 'Site', 'Camera', 'Light', 'Tendon', 'Actuator', 'Constraint', 'Flex', 'Skin', 'Selection', 'SelPoint', 'Contact', 'ContactForce', 'Island')
mjMAXCONPAIR: int = 50
mjMAXFLEXNODES: int = 27
mjMAXIMP: float = 0.9999
mjMAXLIGHT: int = 100
mjMAXLINE: int = 100
mjMAXLINEPNT: int = 1001
mjMAXOVERLAY: int = 500
mjMAXPLANEGRID: int = 200
mjMAXTREEDEPTH: int = 50
mjMAXVAL: float = 10000000000.0
mjMINIMP: float = 0.0001
mjMINMU: float = 1e-05
mjMINVAL: float = 1e-15
mjNBIAS: int = 10
mjNDYN: int = 10
mjNEQDATA: int = 11
mjNFLUID: int = 12
mjNGAIN: int = 10
mjNGROUP: int = 6
mjNIMP: int = 5
mjNISLAND: int = 20
mjNREF: int = 2
mjNSENS: int = 3
mjNSOLVER: int = 200
mjPI: float = 3.141592653589793
mjRNDSTRING: tuple = (('Shadow', '1', 'S'), ('Wireframe', '0', 'W'), ('Reflection', '1', 'R'), ('Additive', '0', 'L'), ('Skybox', '1', 'K'), ('Fog', '0', 'G'), ('Haze', '1', '/'), ('Segment', '0', ','), ('Id Color', '0', ''), ('Cull Face', '1', ''))
mjTIMERSTRING: tuple = ('step', 'forward', 'inverse', 'position', 'velocity', 'actuation', 'constraint', 'advance', 'pos_kinematics', 'pos_inertia', 'pos_collision', 'pos_make', 'pos_project', 'col_broadphase', 'col_narrowphase')
mjVERSION_HEADER: int = 337
mjVISSTRING: tuple = (('Convex Hull', '0', 'H'), ('Texture', '1', 'X'), ('Joint', '0', 'J'), ('Camera', '0', 'Q'), ('Actuator', '0', 'U'), ('Activation', '0', ','), ('Light', '0', 'Z'), ('Tendon', '1', 'V'), ('Range Finder', '1', 'Y'), ('Equality', '0', 'E'), ('Inertia', '0', 'I'), ('Scale Inertia', '0', "'"), ('Perturb Force', '0', 'B'), ('Perturb Object', '1', 'O'), ('Contact Point', '0', 'C'), ('Island', '0', 'N'), ('Contact Force', '0', 'F'), ('Contact Split', '0', 'P'), ('Transparent', '0', 'T'), ('Auto Connect', '0', 'A'), ('Center of Mass', '0', 'M'), ('Select Point', '0', ''), ('Static Body', '1', 'D'), ('Skin', '1', ';'), ('Flex Vert', '0', ''), ('Flex Edge', '1', ''), ('Flex Face', '0', ''), ('Flex Skin', '1', ''), ('Body Tree', '0', '`'), ('Mesh Tree', '0', '\\'), ('SDF iters', '0', ''))
