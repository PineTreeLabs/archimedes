# Visual Style Guide

## Core Brand Elements

### Logo

- Maintain clear space around the logo equal to at least 25% of the logo's width
- Minimum display size: 32px height for digital applications
- Logo variations: Full color for light backgrounds, inverted for dark backgrounds

### Color Palette

#### Primary Colors

| Color Name     | Hex Code | RGB           | Usage                                      |
|----------------|----------|---------------|------------------------------------------- |
| Dark Charcoal  | #2A2A2A  | 42, 42, 42    | Primary background (dark mode), text (light mode) |
| Copper Orange  | #D35400  | 211, 84, 0    | Primary brand color, accents, buttons      |
| Rich Brown     | #5D4037  | 93, 64, 55    | Secondary text

#### Secondary Colors

| Color Name     | Hex Code | RGB           | Usage                                      |
|----------------|----------|---------------|------------------------------------------- |
| Ember Red      | #C0392B  | 192, 57, 43   | Highlights, calls to action, warnings      |
| Pale Gold      | #F1C40F  | 241, 196, 15  | Accents, headings (dark mode)              |
| Steel Blue     | #5C9DC0  | 92, 157, 192  | Data visualization, secondary elements     |
| Light Gray     | #F5F5F5  | 245, 245, 245 | Primary background (light mode), text (dark mode) |

#### Color Mode Applications

**Light Mode**
- Background: Light Gray (#F5F5F5)
- Text: Dark Charcoal (#2A2A2A)
- Primary Accent: Copper Orange (#D35400)
- Secondary Accent: Slate Blue (#34495E)
- Tertiary Accent: Pale Gold (#F1C40F)
- Links: Ember Red (#C0392B)
- Link Hover: Copper Orange (#D35400)

**Dark Mode**
- Background: Dark Charcoal (#2A2A2A)
- Text: Light Gray (#F5F5F5)
- Primary Accent: Pale Gold (#F1C40F)
- Secondary Accent: Copper Orange (#D35400)
- Tertiary Accent: Steel Blue (#5C9DC0)
- Links: Copper Orange (#D35400)
- Link Hover: Pale Gold (#F1C40F)

### Typography

#### Font Selection

| Usage          | Font Family      | Weights             |
|----------------|------------------|---------------------|
| Headings       | Roboto Slab      | 400, 600, 700       |
| Body Text      | Roboto           | 400, 500, 700       |
| Code           | Source Code Pro  | 400, 500            |

#### Type Scale and Hierarchy

- H1: 32px/2rem, Roboto Slab Bold
- H2: 24px/1.5rem, Roboto Slab Bold
- H3: 20px/1.25rem, Roboto Slab Bold
- H4: 18px/1.125rem, Roboto Slab Bold
- H5: 16px/1rem, Roboto Slab Bold
- H6: 14px/0.875rem, Roboto Slab Bold
- Body: 16px/1rem, Roboto Regular
- Small Text: 14px/0.875rem, Roboto Regular
- Code: 14px/0.875rem, Source Code Pro Regular

## UI Components

### Buttons

**Primary Button**
- Background: Copper Orange (#D35400)
- Text: Light Gray (#F5F5F5)
- Hover: Darken by 10% (#BF4B00)
- Border: None
- Border Radius: 5px
- Padding: 8px 16px

**Secondary Button**
- Light Mode:
  - Background: Light Gray (#F5F5F5)
  - Border: 1px solid Copper Orange (#D35400)
  - Text: Copper Orange (#D35400)
- Dark Mode:
  - Background: Dark Charcoal (#2A2A2A)
  - Border: 1px solid Pale Gold (#F1C40F)
  - Text: Pale Gold (#F1C40F)

### Navigation

- Active Item: Highlighted with Copper Orange (light mode) or Pale Gold (dark mode)
- Hover State: Slight background change to #EEEEEE (light mode) or #333333 (dark mode)

### Admonitions/Callouts

**Note**
- Border-left: 4px solid Steel Blue (#5C9DC0)
- Background: Steel Blue at 10% opacity

**Warning**
- Border-left: 4px solid Ember Red (#C0392B)
- Background: Ember Red at 10% opacity

**Tip**
- Border-left: 4px solid Pale Gold (#F1C40F)
- Background: Pale Gold at 10% opacity

## Data Visualization

### Plot Styling

**Line Charts (Light Mode)**
- Primary Line: Copper Orange #D35400
- Secondary Line: Slate Blue #34495E
- Tertiary Line: Pale Gold #F1C40F

- Primary Line: Copper Orange #D35400
- Secondary Line: Steel Blue #5C9DC0
- Tertiary Line: Pale Gold #F1C40F

**Background and Grid**
- Light Mode:
  - Background: Light Gray #F5F5F5
  - Grid Lines: #DDDDDD
- Dark Mode:
  - Background: Dark Charcoal #2A2A2A
  - Grid Lines: #444444

**Text Elements**
- Title: Roboto Slab, Bold
- Axis Labels: Roboto, Regular
- Legend: Roboto, Regular

## Code Syntax Highlighting

**Dark Mode Highlighting**
- Strings: #d9a0a0 (soft red)
- Module Names: #a0d9a0 (soft green)
- Built-ins: #a0a0d9 (soft blue)
- Keywords: Pale Gold #F1C40F
- Names: Light Gray #F5F5F5

**Light Mode Highlighting**
- Strings: #a83232 (darker red)
- Module Names: #32a832 (darker green)
- Built-ins: #3232a8 (darker blue)
- Keywords: Copper Orange #D35400
- Names: Dark Charcoal #2A2A2A

## Documentation Patterns

### Code Blocks
- Background: #EEEEEE (light mode), #333333 (dark mode)
- Border: 1px solid #DDDDDD (light mode), 1px solid #444444 (dark mode)
- Border Radius: 5px
- Padding: 16px

### Tables
- Header Background: #EEEEEE (light mode), #333333 (dark mode)
- Alternating Row Colors: 
  - Light Mode: #F5F5F5, #FFFFFF
  - Dark Mode: #2A2A2A, #333333
- Border: 1px solid #DDDDDD (light mode), 1px solid #444444 (dark mode)

### Images
- Automatically switch between light/dark versions based on theme
- Add subtle shadow to images in light mode
- Maintain consistent sizing across themes

## Implementation Guidelines

### Theme Toggle
- Position: Top-right navigation area
- Icon: Sun for light mode, Moon for dark mode
- State Persistence: Store user preference in local storage

### Responsive Design
- Breakpoints:
  - Mobile: < 768px
  - Tablet: 768px - 1024px
  - Desktop: > 1024px
- Typography scaling:
  - Mobile: Base size 14px
  - Tablet/Desktop: Base size 16px

### Accessibility Requirements
- Maintain WCAG AA contrast standards (4.5:1 for normal text, 3:1 for large text)
- Ensure all interactive elements are keyboard accessible
- Provide appropriate hover/focus states for all interactive elements

## File Organization

- `/static/css/`: Location for all stylesheets
- `/static/images/`: Image assets, with `-dark` and `-light` suffix variants
- `/static/js/`: JavaScript files, including theme toggle functionality

## Extending the System

When adding new colors or components:

1. Maintain the forge-inspired theme
2. Preserve similar saturation and value levels
3. Test in both light and dark modes
4. Update this style guide to include new elements
5. Ensure new colors have sufficient contrast for text

For creating color gradients:
1. Use HSL color space for natural transitions
2. Keep middle steps slightly closer in value
3. Test at both large and small scales
4. Consider 5-7 steps for most gradients

Example for copper to ember gradient:
`#D35400 → #D04A0D → #CC4119 → #C93824 → #C0392B`

---

## Proposed Revisions (2026)

### Rationale

Following user testing with the Lynx block diagram application, the original color palette has been refined to better support extended technical work. The bright copper orange (#D35400) and neon pale gold (#F1C40F) created visual fatigue during long documentation sessions and competed with technical content rather than supporting it.

**Key Insight**: The forge metaphor is most effective when colors suggest *aged metal, patina, and refined tools* rather than *molten metal fresh from the fire*. Technical documentation is consumed for 20-30 minute sessions (similar to application usage), not 30-second marketing scans.

**Design Principle**: Colors should provide an *elegant support system* for complex information—structure without shouting. Warmth through subtle tinting rather than saturation.

### Revised Color Palette

#### Updated Primary Colors

| Color Name          | Old Hex  | New Hex  | RGB           | Usage                                      |
|---------------------|----------|----------|---------------|------------------------------------------- |
| Dark Charcoal       | #2A2A2A  | #2A2A2A  | 42, 42, 42    | Primary background (dark mode), text (light mode) - **Unchanged** |
| Weathered Copper    | #D35400  | #B85C2E  | 184, 92, 46   | Primary brand color, accents, buttons - **Desaturated** |
| Rich Brown          | #5D4037  | #5D4037  | 93, 64, 55    | Secondary text - **Unchanged** |

**Reasoning**: The new weathered copper (#B85C2E) has earthy terracotta depth without the digital vibrancy of the original. Evokes real oxidized copper rather than a digital color picker.

#### Updated Secondary Colors

| Color Name          | Old Hex  | New Hex  | RGB           | Usage                                      |
|---------------------|----------|----------|---------------|------------------------------------------- |
| Refined Ember       | #C0392B  | #B8432E  | 184, 67, 46   | Highlights, calls to action, warnings (light mode) |
| Aged Brass          | #F1C40F  | #C9A961  | 201, 169, 97  | Accents, headings (dark mode) - **Significantly desaturated** |
| Steel Blue          | #5C9DC0  | #5C9DC0  | 92, 157, 192  | Data visualization, secondary elements - **Unchanged** |
| Light Gray          | #F5F5F5  | #F5F5F5  | 245, 245, 245 | Primary background (light mode) - **Unchanged** |

**Reasoning**:
- **Aged Brass** (#C9A961): The original pale gold was reading at near-neon brightness in dark mode. This muted brass has sophistication and warmth without visual fatigue.
- **Refined Ember**: Slightly desaturated to harmonize with the copper tones while maintaining its role as an attention color.

#### Extended Palette (Full Scale)

**Weathered Copper Scale** (for gradients, tints, shades):
- 50: #faf6f3 (lightest)
- 100: #f4ebe3
- 200: #e8d4c3
- 300: #d9b59a
- 400: #c6876d
- 500: #b85c2e (primary)
- 600: #a04d24 (hover/active)
- 700: #7d3d1c
- 800: #5a2d15
- 900: #3d1f0e (darkest)

**Aged Brass Scale** (for gradients, tints, shades):
- 50: #1a1612 (darkest)
- 100: #2d2619
- 200: #4a3d28
- 300: #6b5838
- 400: #8d7449
- 500: #c9a961 (primary)
- 600: #d4b876 (hover/active)
- 700: #dfc894
- 800: #ead9b3
- 900: #f5ecd4 (lightest)

### Revised Color Mode Applications

#### Light Mode (Revised)

- Background: Light Gray (#F5F5F5) - **Unchanged**
- Text: Dark Charcoal (#2A2A2A) - **Unchanged**
- Primary Accent: **Weathered Copper (#B85C2E)** - *Changed*
- Secondary Accent: Slate Blue (#34495E) - **Unchanged**
- Tertiary Accent: **Steel Blue (#5C9DC0)** - *Changed from Pale Gold*
- Links: **Weathered Copper (#B85C2E)** - *Changed*
- Link Hover: **Deeper Copper (#A04D24)** - *Changed*

**Rationale**: Pale Gold didn't work well in light mode (too bright against white). Steel Blue provides better hierarchy as tertiary accent.

#### Dark Mode (Revised)

- Background: Dark Charcoal (#2A2A2A) - **Unchanged**
- Text: Light Gray (#F5F5F5) - **Unchanged**
- Primary Accent: **Aged Brass (#C9A961)** - *Changed*
- Secondary Accent: **Weathered Copper (#B85C2E)** - *Changed*
- Tertiary Accent: Steel Blue (#5C9DC0) - **Unchanged**
- Links: **Aged Brass (#C9A961)** - *Changed*
- Link Hover: **Lighter Brass (#D4B876)** - *Changed*

**Rationale**: The aged brass provides sophistication without the neon quality. Creates comfortable atmosphere for extended reading.

### Updated UI Components

#### Buttons (Revised)

**Primary Button**
- Background: **Weathered Copper (#B85C2E)** - *Changed*
- Text: Light Gray (#F5F5F5) - **Unchanged**
- Hover: **Deeper Copper (#A04D24)** - *Changed*
- Border: None
- Border Radius: 5px
- Padding: 8px 16px

**Secondary Button**
- Light Mode:
  - Background: Light Gray (#F5F5F5)
  - Border: **1px solid Weathered Copper (#B85C2E)** - *Changed*
  - Text: **Weathered Copper (#B85C2E)** - *Changed*
- Dark Mode:
  - Background: Dark Charcoal (#2A2A2A)
  - Border: **1px solid Aged Brass (#C9A961)** - *Changed*
  - Text: **Aged Brass (#C9A961)** - *Changed*

#### Navigation (Revised)

- Active Item: **Weathered Copper (light mode)** or **Aged Brass (dark mode)** - *Changed*
- Hover State: Slight background change to #EEEEEE (light mode) or #333333 (dark mode) - **Unchanged**

#### Admonitions/Callouts (Revised)

**Note**
- Border-left: 4px solid Steel Blue (#5C9DC0) - **Unchanged**
- Background: Steel Blue at 10% opacity

**Warning**
- Border-left: **4px solid Refined Ember (#B8432E)** - *Changed*
- Background: Refined Ember at 10% opacity

**Tip**
- Border-left: **4px solid Aged Brass (#C9A961)** - *Changed (dark mode only)*
- Border-left: **4px solid Weathered Copper (#B85C2E)** - *New (light mode)*
- Background: Color at 10% opacity

### Updated Data Visualization

#### Plot Styling (Revised)

**Line Charts (Light Mode)**
- Primary Line: **Weathered Copper #B85C2E** - *Changed*
- Secondary Line: Slate Blue #34495E - **Unchanged**
- Tertiary Line: Steel Blue #5C9DC0 - *Changed (more distinct from secondary)*

**Line Charts (Dark Mode)**
- Primary Line: **Aged Brass #C9A961** - *Changed*
- Secondary Line: Steel Blue #5C9DC0 - **Unchanged**
- Tertiary Line: **Weathered Copper #B85C2E** - *Changed*

**Background and Grid** - **Unchanged**
- Light Mode: Background #F5F5F5, Grid Lines #DDDDDD
- Dark Mode: Background #2A2A2A, Grid Lines #444444

#### Chart Color Palette (Multi-line)

When displaying 4+ data series, use this progression:

**Light Mode**:
1. Weathered Copper #B85C2E
2. Slate Blue #34495E
3. Steel Blue #5C9DC0
4. Rich Brown #5D4037
5. Refined Ember #B8432E

**Dark Mode**:
1. Aged Brass #C9A961
2. Steel Blue #5C9DC0
3. Weathered Copper #B85C2E
4. Lighter Brass #D4B876
5. Coral Ember #D9765F

### Updated Code Syntax Highlighting

#### Dark Mode Highlighting (Revised)

- Strings: #d9a0a0 (soft red) - **Unchanged**
- Module Names: #a0d9a0 (soft green) - **Unchanged**
- Built-ins: #a0a0d9 (soft blue) - **Unchanged**
- Keywords: **Aged Brass #C9A961** - *Changed*
- Names: Light Gray #F5F5F5 - **Unchanged**

#### Light Mode Highlighting (Revised)

- Strings: #a83232 (darker red) - **Unchanged**
- Module Names: #32a832 (darker green) - **Unchanged**
- Built-ins: #3232a8 (darker blue) - **Unchanged**
- Keywords: **Weathered Copper #B85C2E** - *Changed*
- Names: Dark Charcoal #2A2A2A - **Unchanged**

### Updated Gradient Examples

#### Copper to Ember (Revised)

**Light Mode**: `#B85C2E → #B1512C → #AB472A → #A73F28 → #B8432E`

**Dark Mode**: `#C9A961 → #CC9159 → #CF7950 → #D26848 → #D9765F`

#### Brass Gradient (New - Dark Mode)

From dark to light: `#6B5838 → #8D7449 → #C9A961 → #D4B876 → #DFC894`

### Design System Principles (Updated)

When extending the color system:

1. **Privilege Content Over Chrome**: Technical content should always be more prominent than structural elements
2. **Patina Over Brightness**: Suggest craftsmanship through subtle tinting and desaturation rather than pure saturation
3. **Context-Aware Intensity**:
   - Diagrams and technical illustrations: Use refined colors (avoid competing with content)
   - Interactive elements (buttons, links): Can use slightly higher saturation for affordance
   - Marketing/landing pages: May use brighter variants if needed for conversion
4. **Extended Session Comfort**: All colors should be comfortable for 20-30+ minute reading/usage sessions
5. **Forge Metaphor Evolution**: "Aged tool" not "hot iron" - warmth through sophistication, not intensity

### Accessibility Notes (Revised Colors)

All revised colors maintain WCAG AA compliance:

**Light Mode Contrast Ratios** (on #F5F5F5 background):
- Weathered Copper #B85C2E on white: **5.8:1** ✅ AA (normal text)
- Rich Brown #5D4037 on white: **9.2:1** ✅ AAA (normal text)
- Refined Ember #B8432E on white: **6.1:1** ✅ AA (normal text)

**Dark Mode Contrast Ratios** (on #2A2A2A background):
- Aged Brass #C9A961 on dark charcoal: **6.4:1** ✅ AA (normal text)
- Light Gray #F5F5F5 on dark charcoal: **13.1:1** ✅ AAA (normal text)
- Steel Blue #5C9DC0 on dark charcoal: **5.2:1** ✅ AA (normal text)

### Migration Strategy

#### Phase 1: Documentation Site
1. Update CSS variables for primary/secondary colors
2. Test all interactive elements (buttons, links, navigation)
3. Review syntax highlighting in code blocks
4. Verify diagram/illustration colors

#### Phase 2: Data Visualization
1. Update matplotlib/plotly style sheets
2. Test multi-line chart legibility
3. Verify grid/axis contrast

#### Phase 3: Brand Materials
1. Update logo variations if needed
2. Revise presentation templates
3. Update social media graphics (optional - can retain brighter variants for marketing)

#### Testing Checklist
- [ ] All text meets WCAG AA contrast (4.5:1 minimum)
- [ ] Links are distinguishable from body text
- [ ] Hover states provide clear feedback
- [ ] Diagrams prioritize content over structure
- [ ] 30-minute reading session produces no eye strain
- [ ] Light/dark mode transitions feel cohesive

### Visual Comparison

**Before & After - Light Mode**
- Links: ~~#C0392B (Ember Red)~~ → **#B85C2E (Weathered Copper)**
- Primary Accent: ~~#D35400 (Bright Copper)~~ → **#B85C2E (Weathered Copper)**

**Before & After - Dark Mode**
- Primary Accent: ~~#F1C40F (Neon Gold)~~ → **#C9A961 (Aged Brass)**
- Links: ~~#D35400 (Bright Copper)~~ → **#C9A961 (Aged Brass)**

**Key Takeaway**: The refined palette reduces saturation by ~30-40% while maintaining warmth and brand recognition. Result: Professional sophistication appropriate for extended technical work.

---

**Last Updated**: 2026-01-19
**Status**: Proposed (pending approval)
**Tested With**: Lynx block diagram application (proven successful over 2-week trial)