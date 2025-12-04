---
title: Distribution
---

::: lerax.distribution.AbstractDistribution
    options:
        members: true

::: lerax.distribution.AbstractMaskableDistribution
    options:
        members: ["mask"]

::: lerax.distribution.AbstractTransformedDistribution
    options:
        members: ["distribution", "bijector"]
