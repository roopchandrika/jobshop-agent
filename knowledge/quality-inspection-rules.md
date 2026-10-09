# Quality: inspection rules

## Housings
Every housing order needs a first-article inspection after its grinding step. The inspection takes 15 minutes
and blocks the order from moving on. The scheduler does not model this inspection time, so planners should
allow for it by hand.

## Gears
Gear orders need a tooth-profile check after milling. A failed check sends the order back to milling.

## Quality holds
An order on quality hold must not be shipped even if it is finished on time. Holds are placed only by the quality
manager.
