ament_prepend_unique_value GZ_SIM_RESOURCE_PATH "$AMENT_CURRENT_PREFIX/share/wamv_description/models"
# 让 gz 能解析 model://wamv_description/... (share 下需有 wamv_description 子目录)
ament_prepend_unique_value GZ_SIM_RESOURCE_PATH "$AMENT_CURRENT_PREFIX/share"
