from pyk4a import PyK4A, connected_device_count


cnt = connected_device_count()
if not cnt:
    print("No devices available")
    exit()
id_dict = {}
print(f"Available devices: {cnt}")
for device_id in range(cnt):
    device = PyK4A(device_id=device_id)
    device.open()
    print(device_id, device.serial)
    id_dict.update({device.serial:device_id})
    print(id_dict)
    device.close()
