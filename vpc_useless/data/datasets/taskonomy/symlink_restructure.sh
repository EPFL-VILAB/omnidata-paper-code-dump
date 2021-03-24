export SRC="${1}"
export TARGET="${2}"
export SCRIPT_DIR=$(dirname $(realpath -s $0)) 


[ -z "$SRC" ] && { echo "symlink_restructure.sh: Variable 'SRC' (arg0) must be nonempty!"; exit 1; }
[ -z "$TARGET" ] && { echo "Variable 'TARGET' (arg1) must be nonempty!"; exit 1; }


for task in rgb class_object class_scene depth_euclidean depth_zbuffer edge_occlusion edge_texture keypoints2d keypoints3d nonfixated_matches normal point_info principal_curvature segment_semantic segment_unsup25d segment_unsup2d reshading; do
    mkdir -p "${TARGET}/$task"
    for model in $(python ${SCRIPT_DIR}/print_splits.py fullplus); do
        ln -s "${SRC}/${model}_${task}/${task}/" "${TARGET}/$task/$model"
    done
done

mkdir -p "${TARGET}/mask_valid"
for model in $(python ${SCRIPT_DIR}/print_splits.py fullplus); do
    ln -s "/datasets/taskonomymask/${model}_mask_valid/" "${TARGET}/mask_valid/$model"
done