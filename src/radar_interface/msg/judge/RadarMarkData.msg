# RoboMaster 2026 protocol 0x020C: bit field.
# bit 0: opponent hero vulnerability mark
# bit 1: opponent engineer vulnerability mark
# bit 2: opponent infantry 3 vulnerability mark
# bit 3: opponent infantry 4 vulnerability mark
# bit 4: opponent aerial special mark
# bit 5: opponent sentry vulnerability mark
uint16 mark_progress

uint16 HERO_MASK = 1
uint16 ENGINEER_MASK = 2
uint16 INFANTRY_3_MASK = 4
uint16 INFANTRY_4_MASK = 8
uint16 AERIAL_MASK = 16
uint16 SENTRY_MASK = 32
