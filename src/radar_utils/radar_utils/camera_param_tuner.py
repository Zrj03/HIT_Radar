import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, Optional

import rclpy
from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    minimum: float
    maximum: float
    step: float
    decimals: int
    default: float


class CameraParamTunerUI:
    PARAMS = (
        ParamSpec("exposure_time", "曝光时间", 1000.0, 40000.0, 100.0, 1, 12000.0),
        ParamSpec("gain", "增益", 0.0, 64.0, 0.5, 2, 23.0),
        ParamSpec("digital_shift", "数字增益", 0.0, 16.0, 0.1, 2, 1.0),
    )

    def __init__(self, node: Node):
        self.node = node

        self.root = tk.Tk()
        self.root.title("相机参数调节")
        self.root.geometry("980x340")
        self.root.resizable(True, True)
        self.root.minsize(820, 280)

        self._controls: Dict[str, Dict[str, object]] = {}
        self._programmatic_update = False
        self._pending_clients = []

        self.target_node_var = tk.StringVar(value="/radar/hik_6mm/hik_camera")
        self.status_var = tk.StringVar(value="就绪")

        self._build_ui()

        self.spin_thread_stop = threading.Event()
        self.spin_thread = threading.Thread(target=self._spin_worker, daemon=True)
        self.spin_thread.start()

        self.root.after(200, self.refresh_from_node)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="目标节点:").pack(side=tk.LEFT)
        target_entry = ttk.Entry(top, textvariable=self.target_node_var, width=38)
        target_entry.pack(side=tk.LEFT, padx=(8, 8))

        ttk.Button(top, text="读取当前", command=self.refresh_from_node).pack(side=tk.LEFT)
        ttk.Button(top, text="应用全部", command=self.apply_all).pack(side=tk.LEFT, padx=(8, 0))

        body = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        body.pack(fill=tk.BOTH, expand=True)

        for row_idx, spec in enumerate(self.PARAMS):
            ttk.Label(body, text=spec.label, width=16).grid(row=row_idx, column=0, sticky=tk.W, padx=(0, 8), pady=8)

            value_var = tk.DoubleVar(value=spec.default)
            scale = tk.Scale(
                body,
                from_=spec.minimum,
                to=spec.maximum,
                orient=tk.HORIZONTAL,
                resolution=spec.step,
                showvalue=False,
                length=560,
                variable=value_var,
                command=lambda _val, k=spec.key: self._on_scale_change(k),
            )
            scale.grid(row=row_idx, column=1, sticky=tk.EW, padx=(0, 8), pady=4)
            scale.bind("<ButtonRelease-1>", lambda _e, k=spec.key: self.apply_one(k))

            entry = ttk.Entry(body, width=12)
            entry.grid(row=row_idx, column=2, sticky=tk.W, padx=(0, 6), pady=4)
            entry.insert(0, self._fmt(spec, spec.default))
            entry.bind("<Return>", lambda _e, k=spec.key: self._on_entry_commit(k))
            entry.bind("<FocusOut>", lambda _e, k=spec.key: self._on_entry_commit(k))

            plus_btn = ttk.Button(body, text="+", width=3, command=lambda k=spec.key: self._nudge(k, +1))
            plus_btn.grid(row=row_idx, column=3, sticky=tk.W, padx=(0, 4))

            minus_btn = ttk.Button(body, text="-", width=3, command=lambda k=spec.key: self._nudge(k, -1))
            minus_btn.grid(row=row_idx, column=4, sticky=tk.W)

            self._controls[spec.key] = {
                "spec": spec,
                "var": value_var,
                "scale": scale,
                "entry": entry,
            }

        body.columnconfigure(1, weight=1)

        status = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)

    def _spec(self, key: str) -> ParamSpec:
        return self._controls[key]["spec"]  # type: ignore[return-value]

    def _var(self, key: str) -> tk.DoubleVar:
        return self._controls[key]["var"]  # type: ignore[return-value]

    def _entry(self, key: str) -> ttk.Entry:
        return self._controls[key]["entry"]  # type: ignore[return-value]

    def _fmt(self, spec: ParamSpec, value: float) -> str:
        return f"{value:.{spec.decimals}f}"

    def _clamp(self, spec: ParamSpec, value: float) -> float:
        value = max(spec.minimum, min(spec.maximum, value))
        # Round to configured precision so slider/entry stay aligned.
        return round(value, spec.decimals)

    def _set_control_value(self, key: str, value: float) -> None:
        spec = self._spec(key)
        value = self._clamp(spec, value)
        self._programmatic_update = True
        try:
            self._var(key).set(value)
            e = self._entry(key)
            e.delete(0, tk.END)
            e.insert(0, self._fmt(spec, value))
        finally:
            self._programmatic_update = False

    def _get_control_value(self, key: str) -> float:
        return float(self._var(key).get())

    def _on_scale_change(self, key: str) -> None:
        if self._programmatic_update:
            return
        spec = self._spec(key)
        value = self._clamp(spec, self._get_control_value(key))
        self._set_control_value(key, value)

    def _on_entry_commit(self, key: str) -> None:
        if self._programmatic_update:
            return
        spec = self._spec(key)
        text = self._entry(key).get().strip()
        try:
            value = float(text)
        except ValueError:
            self.status_var.set(f"{spec.label} 输入无效: {text}")
            self._set_control_value(key, self._get_control_value(key))
            return

        self._set_control_value(key, value)
        self.apply_one(key)

    def _nudge(self, key: str, direction: int) -> None:
        spec = self._spec(key)
        current = self._get_control_value(key)
        self._set_control_value(key, current + direction * spec.step)
        self.apply_one(key)

    def _spin_worker(self) -> None:
        while rclpy.ok() and not self.spin_thread_stop.is_set():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except ExternalShutdownException:
                break

    def _set_status(self, text: str) -> None:
        self.root.after(0, lambda: self.status_var.set(text))

    def _target_node(self) -> str:
        target = self.target_node_var.get().strip()
        return target if target else "/radar/hik_6mm/hik_camera"

    def refresh_from_node(self) -> None:
        target = self._target_node()
        client = self.node.create_client(GetParameters, f"{target}/get_parameters")
        if not client.wait_for_service(timeout_sec=0.3):
            self._set_status(f"服务不可用: {target}/get_parameters")
            return
        self._pending_clients.append(client)
        names = [p.key for p in self.PARAMS]
        req = GetParameters.Request()
        req.names = names
        fut = client.call_async(req)

        def _done_cb(done_fut):
            try:
                values = done_fut.result()
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"读取参数失败 ({target}): {exc}")
                if client in self._pending_clients:
                    self._pending_clients.remove(client)
                return

            for spec, val in zip(self.PARAMS, values.values):
                if val.type == ParameterType.PARAMETER_NOT_SET:
                    continue
                if val.type == ParameterType.PARAMETER_DOUBLE:
                    v = float(val.double_value)
                elif val.type == ParameterType.PARAMETER_INTEGER:
                    v = float(val.integer_value)
                else:
                    continue
                self.root.after(0, lambda k=spec.key, vv=v: self._set_control_value(k, vv))

            self._set_status(f"已读取参数: {target}")
            if client in self._pending_clients:
                self._pending_clients.remove(client)

        fut.add_done_callback(_done_cb)

    def _set_param(self, key: str, value: float, done_text: str) -> None:
        target = self._target_node()
        client = self.node.create_client(SetParameters, f"{target}/set_parameters")
        if not client.wait_for_service(timeout_sec=0.3):
            self._set_status(f"服务不可用: {target}/set_parameters")
            return
        self._pending_clients.append(client)
        param = Parameter(name=key, value=value)
        req = SetParameters.Request()
        req.parameters = [param.to_parameter_msg()]
        fut = client.call_async(req)

        def _done_cb(done_fut):
            try:
                results = done_fut.result().results
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"设置 {key} 失败: {exc}")
                if client in self._pending_clients:
                    self._pending_clients.remove(client)
                return
            if results and results[0].successful:
                self._set_status(done_text)
            else:
                reason = results[0].reason if results else "Unknown"
                self._set_status(f"设置 {key} 被拒绝: {reason}")
            if client in self._pending_clients:
                self._pending_clients.remove(client)

        fut.add_done_callback(_done_cb)

    def apply_one(self, key: str) -> None:
        value = self._get_control_value(key)
        spec = self._spec(key)
        value = self._clamp(spec, value)
        self._set_control_value(key, value)
        self._set_param(key, value, f"已应用 {spec.label}={self._fmt(spec, value)}")

    def apply_all(self) -> None:
        target = self._target_node()
        client = self.node.create_client(SetParameters, f"{target}/set_parameters")
        if not client.wait_for_service(timeout_sec=0.3):
            self._set_status(f"服务不可用: {target}/set_parameters")
            return
        self._pending_clients.append(client)

        params = []
        for spec in self.PARAMS:
            value = self._clamp(spec, self._get_control_value(spec.key))
            self._set_control_value(spec.key, value)
            params.append(Parameter(name=spec.key, value=value).to_parameter_msg())

        req = SetParameters.Request()
        req.parameters = params
        fut = client.call_async(req)

        def _done_cb(done_fut):
            try:
                results = done_fut.result().results
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"应用全部参数失败: {exc}")
                if client in self._pending_clients:
                    self._pending_clients.remove(client)
                return

            if all(r.successful for r in results):
                self._set_status(f"已应用全部参数到: {target}")
                if client in self._pending_clients:
                    self._pending_clients.remove(client)
                return

            failures = [r.reason for r in results if not r.successful]
            self._set_status("部分参数应用失败: " + "; ".join(failures))
            if client in self._pending_clients:
                self._pending_clients.remove(client)

        fut.add_done_callback(_done_cb)

    def _on_close(self) -> None:
        self.spin_thread_stop.set()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = Node("camera_param_tuner")

    ui = CameraParamTunerUI(node)
    try:
        ui.run()
    finally:
        ui.spin_thread_stop.set()
        if ui.spin_thread.is_alive():
            ui.spin_thread.join(timeout=1.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
