1. 配置文件中 `palm` 字段下 `compatible_bbox_expand` 字段的含义
2. `01_export_palm_detections.py` 生成的 palm detection 是如何通过运算变为 Hand ROI 的？Hand ROI 中是否会有因为旋转/平移等操作造成的 crop 外的黑色区域？如果有，如何处理这些黑色区域？这些黑色区域是否会作为 crop 图片的一部分直接参与训练和推理？这种处理方式必须与板端调度程序保持一致！
3. `02_build_hand_roi_crops.py` 生成的中间标注文件中，每一项的字段 `roi_rect/roi_corners_px` 的含义，如何使用它来反投影 MediaPipe 输出的 crop 坐标到原图坐标。这一部分详细解释一下，可以用数学公式
4. `03_run_mediapipe_on_rois.py` 生成的两份：`data/labels/hand_landmarks_mediapipe_raw.jsonl` 和 `data/labels/hand_landmarks_autolabel_draft.jsonl` 分别是什么意思？我看这两份文件完全相同，可否只保留一份，保持简洁？`hand_presence.present` 字段的值是否完全取决于 `palm_valid`
5. 整套系统中，1个 crop_id 是否只有唯一的 palm_det_id 前缀？并且一个 palm_det_id 是否只会生产出一个 crop_id ？同理， 1 个 crop_id 是否只可能生产出一个唯一的 hand_id（当然也可能没有）？如果是这样的话，那么 crop_id 在 palm_det_id 后面添加的后缀就不要用 `:crop0` 了，直接用 `:crop` 就可以了。同理， hand_id 在 crop_id 后面添加的后缀也可以直接用 `:hand` 就可以了。
6. `04_export_cvat_xml.py` 脚本生成的 `data/review/cvat_upload_images/` 与 `data/roi_crops/images/*.png` 是否完全一致？如果一致，请不要重新复制一份，因为磁盘空间/服务器空间有限，我直接把 `data/roi_crops/images/*.png` 上传到 CVAT 就可以了。
7. 关于 cvat 网站的 .xml 转化以及上传，我还有一个问题：我们是如何处理负样本的？也就是 `palm_valid=false`/`hand_presence.present=false` 的样本也会被全部上传到 CVAT 进行标注，如果人工复核出该负样本确实没有手，无法标注，那么这种样本如何处理？如果人工复核出反而有手，又如何处理？以及就算确实 `palm_valid=true`/`hand_presence.present=true`，但是人工复核出确实没有手，那么这种样本如何处理？
8. 对于 7. 中提到的这几类异常的样本，它们如何参与训练，需要参与训练吗？请对每一类样本进行详细的解释。
9. 请将 data 的输出结构调整为如下格式
    ```
    ├─ data/
    │  ├─ images/                # 原始图片
    │  ├─ 01_palm/               # Palm 检测结果
    │  ├─ 02_roi_crops/          # ROI crop 图片、Mediapipe Hand Landmark 标注初稿、转化的 xml
    │  ├─ 03_reviewd/            # 人工复核结果
    │  ├─ 04_visualization/      # 可视化结果
    │  ├─ review/                # CVAT 上传、复核和可视化
    │  └─ qc/                    # QC 报告
    ```

    - `01_palm/` 放：
        - `palm_detections.jsonl`
    - `02_roi_crops/` 放：
        - `02_build_hand_roi_crops.py` 输出的：`images/*.png`, `hand_roi_crops_manifest.jsonl`
        - `03_run_mediapipe_on_rois.py` 输出的：`hand_landmarks_mediapipe_raw.jsonl`
        - `04_export_cvat_xml.py` 输出的：`cvat_autolabel.xml`
    - `03_reviewed` 放：
        - 人工标注的：`cvat_reviewed.xml`
        - `05_import_cvat_xml.py` 输出的：`hand_landmarks_reviewed.jsonl`
    - `04_visualization` 放：
        - `crop_images/`: 在 crop 小图上进行标注可视化的输出图片（一张图最多一只手）
        - `global_images/`: 在原图上进行标注可视化的输出图片
        - 其余你认为必要的文件
    - `05_labels` 放：
        - `07_finalize_training_labels.py` 输出的：`hand_training_labels.jsonl`
    
    注意，只调整输出文件的存放结构，不改变内容、也不改变 `qc\` 下的报告结构。