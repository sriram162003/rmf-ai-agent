"""Tests for the FleetRegistry."""

import asyncio
import pytest
from rmf.fleet_scanner import FleetRegistry, FleetEntry, RobotInfo


@pytest.mark.asyncio
async def test_upsert_new_fleet():
    registry = FleetRegistry()
    entry = FleetEntry(name="fleet-a", robots={"bot-1": RobotInfo(name="bot-1")})
    await registry.upsert(entry)
    snapshot = await registry.snapshot()
    assert "fleet-a" in snapshot
    assert "bot-1" in snapshot["fleet-a"]["robots"]


@pytest.mark.asyncio
async def test_upsert_merges_robots():
    registry = FleetRegistry()
    await registry.upsert(FleetEntry(name="fleet-a", robots={"bot-1": RobotInfo(name="bot-1")}))
    await registry.upsert(FleetEntry(name="fleet-a", robots={"bot-2": RobotInfo(name="bot-2")}))
    snapshot = await registry.snapshot()
    robots = snapshot["fleet-a"]["robots"]
    assert "bot-1" in robots
    assert "bot-2" in robots


@pytest.mark.asyncio
async def test_upsert_merges_capabilities():
    registry = FleetRegistry()
    await registry.upsert(FleetEntry(name="fleet-a", capabilities=["patrol"]))
    await registry.upsert(FleetEntry(name="fleet-a", capabilities=["delivery"]))
    snapshot = await registry.snapshot()
    caps = set(snapshot["fleet-a"]["capabilities"])
    assert "patrol" in caps
    assert "delivery" in caps


@pytest.mark.asyncio
async def test_fleet_names():
    registry = FleetRegistry()
    await registry.upsert(FleetEntry(name="fleet-x"))
    await registry.upsert(FleetEntry(name="fleet-y"))
    assert set(registry.fleet_names()) == {"fleet-x", "fleet-y"}
