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