"""
Fleet discovery via three parallel channels:
  1. RMF API polling (primary)
  2. WiFi / mDNS (Zeroconf) — discovers RMF servers on the LAN
  3. Bluetooth BLE — discovers robots advertising fleet info
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Service type advertised by RMF fleet adapters over mDNS
_MDNS_SERVICE_TYPE = "_rmf-fleet._tcp.local."
# BLE Service UUID used by robots to advertise fleet membership
_DEFAULT_BLE_UUID = "12345678-1234-1234-1234-123456789abc"


@dataclass
class RobotInfo:
    name: str
    model: str = ""
    mode: str = "unknown"
    battery_percent: float = -1.0
    location: dict = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)


@dataclass
class FleetEntry:
    name: str
    robots: dict[str, RobotInfo] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    source: str = "rmf"   # rmf | wifi | ble
    api_url: str = ""


class FleetRegistry:
    """Thread-safe in-memory registry of discovered fleets."""

    def __init__(self) -> None:
        self._fleets: dict[str, FleetEntry] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, entry: FleetEntry) -> None:
        async with self._lock:
            existing = self._fleets.get(entry.name)
            if existing:
                existing.robots.update(entry.robots)
                existing.capabilities = list(
                    set(existing.capabilities) | set(entry.capabilities)
                )
                if entry.api_url:
                    existing.api_url = entry.api_url
            else:
                self._fleets[entry.name] = entry

    async def snapshot(self) -> dict[str, dict]:
        async with self._lock:
            return {
                name: {
                    "robots": {
                        r_name: vars(r) for r_name, r in entry.robots.items()
                    },
                    "capabilities": entry.capabilities,
                    "source": entry.source,
                    "api_url": entry.api_url,
                }
                for name, entry in self._fleets.items()
            }

    def fleet_names(self) -> list[str]:
        return list(self._fleets.keys())


class FleetScanner:
    """
    Runs three concurrent discovery loops.
    Results are merged into a shared FleetRegistry.
    """

    def __init__(
        self,
        rmf_client,  # rmf.client.RMFClient
        registry: FleetRegistry,
        rmf_poll_interval: int = 10,
        wifi_mdns: bool = True,
        bluetooth: bool = True,
        ble_service_uuid: str = _DEFAULT_BLE_UUID,
        ble_scan_duration: float = 5.0,
    ) -> None:
        self._rmf = rmf_client
        self.registry = registry
        self._rmf_interval = rmf_poll_interval
        self._wifi_enabled = wifi_mdns
        self._ble_enabled = bluetooth
        self._ble_uuid = ble_service_uuid
        self._ble_duration = ble_scan_duration
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Launch all discovery loops as background tasks."""
        self._tasks.append(asyncio.create_task(self._rmf_loop(), name="rmf-scanner"))
        if self._wifi_enabled:
            self._tasks.append(asyncio.create_task(self._mdns_loop(), name="mdns-scanner"))
        if self._ble_enabled:
            self._tasks.append(asyncio.create_task(self._ble_loop(), name="ble-scanner"))
        logger.info(
            "FleetScanner started (RMF poll=%ds, WiFi=%s, BLE=%s)",
            self._rmf_interval,
            self._wifi_enabled,
            self._ble_enabled,
        )

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def scan_once(self) -> dict[str, dict]:
        """Trigger a one-shot RMF API scan and return current registry snapshot."""
        await self._rmf_scan()
        return await self.registry.snapshot()

    # ------------------------------------------------------------------
    # Channel 1: RMF REST API polling
    # ------------------------------------------------------------------

    async def _rmf_loop(self) -> None:
        while True:
            try:
                await self._rmf_scan()
            except Exception as exc:
                logger.warning("RMF fleet scan error: %s", exc)
            await asyncio.sleep(self._rmf_interval)

    async def _rmf_scan(self) -> None:
        fleets = await self._rmf.get_fleets()
        for fleet_data in fleets:
            fleet_name = fleet_data.get("name", "")
            if not fleet_name:
                continue
            try:
                state = await self._rmf.get_fleet_state(fleet_name)
            except Exception:
                state = {}

            robots: dict[str, RobotInfo] = {}
            for r in state.get("robots", {}).values() if isinstance(state.get("robots"), dict) else []:
                rname = r.get("name", "")
                if rname:
                    robots[rname] = RobotInfo(
                        name=rname,
                        model=r.get("model", ""),
                        mode=str(r.get("mode", {}).get("mode", "unknown")),
                        battery_percent=r.get("battery_percent", -1.0),
                        location=r.get("location", {}),
                    )

            entry = FleetEntry(
                name=fleet_name,
                robots=robots,
                source="rmf",
                api_url=self._rmf.api_url,
            )
            await self.registry.upsert(entry)
            logger.debug("RMF scan: fleet=%s robots=%d", fleet_name, len(robots))

    # ------------------------------------------------------------------
    # Channel 2: WiFi mDNS / Zeroconf
    # ------------------------------------------------------------------

    async def _mdns_loop(self) -> None:
        try:
            from zeroconf import ServiceBrowser, ServiceStateChange, Zeroconf
            from zeroconf.asyncio import AsyncServiceBrowser, AsyncZeroconf
        except ImportError:
            logger.warning("zeroconf not installed — WiFi fleet discovery disabled")
            return

        azc = AsyncZeroconf()
        discovered: dict[str, dict] = {}

        def on_service_state_change(zeroconf, service_type, name, state_change):
            if state_change == ServiceStateChange.Added:
                asyncio.create_task(self._resolve_mdns(zeroconf, service_type, name, discovered))

        browser = AsyncServiceBrowser(
            azc.zeroconf, _MDNS_SERVICE_TYPE, handlers=[on_service_state_change]
        )
        logger.info("mDNS browser started on service type: %s", _MDNS_SERVICE_TYPE)

        try:
            # Keep alive; re-register every 60s to pick up new services
            while True:
                await asyncio.sleep(60)
        finally:
            await browser.async_cancel()
            await azc.async_close()

    async def _resolve_mdns(
        self,
        zeroconf,
        service_type: str,
        name: str,
        discovered: dict,
    ) -> None:
        try:
            from zeroconf import ServiceInfo

            info = ServiceInfo(service_type, name)
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: info.request(zeroconf, 3000)
            )
            if not info:
                return

            addresses = info.parsed_addresses()
            port = info.port
            fleet_name = info.properties.get(b"fleet", b"").decode("utf-8") or name.split(".")[0]
            api_url = f"http://{addresses[0]}:{port}" if addresses else ""

            entry = FleetEntry(name=fleet_name, source="wifi", api_url=api_url)
            await self.registry.upsert(entry)
            logger.info("mDNS discovered fleet: %s at %s", fleet_name, api_url)
        except Exception as exc:
            logger.debug("mDNS resolve error for %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Channel 3: Bluetooth BLE
    # ------------------------------------------------------------------

    async def _ble_loop(self) -> None:
        try:
            from bleak import BleakScanner
        except ImportError:
            logger.warning("bleak not installed — BLE fleet discovery disabled")
            return

        logger.info("BLE scanner started (service UUID: %s)", self._ble_uuid)
        while True:
            try:
                await self._ble_scan()
            except Exception as exc:
                logger.warning("BLE scan error: %s", exc)
            await asyncio.sleep(self._ble_duration + 2)

    async def _ble_scan(self) -> None:
        from bleak import BleakScanner

        devices = await BleakScanner.discover(
            timeout=self._ble_duration,
            service_uuids=[self._ble_uuid],
        )
        for device in devices:
            try:
                payload = self._parse_ble_advertisement(device)
                if not payload:
                    continue
                fleet_name = payload.get("fleet", device.name or device.address)
                robot_name = payload.get("robot", device.address)
                robot_info = RobotInfo(
                    name=robot_name,
                    model=payload.get("model", ""),
                    battery_percent=float(payload.get("battery", -1)),
                )
                entry = FleetEntry(
                    name=fleet_name,
                    robots={robot_name: robot_info},
                    source="ble",
                )
                await self.registry.upsert(entry)
                logger.info("BLE discovered robot: %s in fleet: %s", robot_name, fleet_name)
            except Exception as exc:
                logger.debug("BLE parse error for %s: %s", device.address, exc)

    @staticmethod
    def _parse_ble_advertisement(device) -> dict | None:
        """
        Parse BLE advertisement data.
        Expected format: JSON string in manufacturer data or service data.
        Example payload: {"fleet": "delivery-fleet", "robot": "bot-01", "battery": 85}
        """
        # Try service data first
        if hasattr(device, "metadata") and device.metadata:
            svc_data = device.metadata.get("uuids", {})
            for uuid, data in (device.metadata.get("service_data") or {}).items():
                try:
                    return json.loads(bytes(data).decode("utf-8"))
                except Exception:
                    pass

            # Try manufacturer data
            for mfr_id, data in (device.metadata.get("manufacturer_data") or {}).items():
                try:
                    return json.loads(bytes(data).decode("utf-8"))
                except Exception:
                    pass

        # Try advertisement data from bleak AdvertisementData
        adv = getattr(device, "details", {})
        if isinstance(adv, dict):
            for key in ("service_data", "manufacturer_data"):
                for uuid_or_id, data in (adv.get(key) or {}).items():
                    try:
                        return json.loads(bytes(data).decode("utf-8"))
                    except Exception:
                        pass
        return None
