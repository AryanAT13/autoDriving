"""Every reward term in one place, so the incentive structure is auditable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepEvents:
    speed_level: int = 0
    lane_changed: bool = False
    off_road: bool = False
    ran_red: bool = False
    collision: bool = False
    goal: bool = False


def step_reward(cfg, events: StepEvents) -> float:
    reward = cfg.progress * events.speed_level + cfg.step_penalty
    if events.lane_changed:
        reward += cfg.lane_change
    if events.off_road:
        reward += cfg.off_road
    if events.ran_red:
        reward += cfg.red_light
    if events.collision:
        reward += cfg.collision
    if events.goal:
        reward += cfg.goal
    return reward
