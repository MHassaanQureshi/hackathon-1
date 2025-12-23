# Docusaurus UI Redesign - Component Documentation

This document provides guidance on using the new UI components and design elements implemented as part of the Docusaurus UI redesign.

## Available Components

### 1. Tip Component
Use for highlighting important information, notes, warnings, or tips.

```md
import Tip from '@site/src/components/Tip';

<Tip>
This is a regular tip for highlighting important information.
</Tip>

<Tip type="note">
This is a note with special styling.
</Tip>

<Tip type="caution">
This is a caution for important warnings.
</Tip>

<Tip type="danger">
This is a danger alert for critical warnings.
</Tip>
```

### 2. Callout Component
Use for creating highlighted sections with optional titles.

```md
import Callout from '@site/src/components/Callout';

<Callout>
This is a default callout for highlighting important content.
</Callout>

<Callout type="info">
This is an info callout for informational content.
</Callout>

<Callout type="warning" title="Warning">
This is a warning callout with a title.
</Callout>
```

### 3. Card Component
Use for organizing content into distinct sections.

```md
import Card from '@site/src/components/Card';

<Card title="Section Title">
This is content inside a card component.
</Card>

<Card type="info" title="Information Card">
This is an info card with special styling.
</Card>
```

### 4. Button Component
Use for creating consistently styled buttons.

```md
import Button from '@site/src/components/Button';

<Button href="/link" type="primary">Primary Button</Button>
<Button type="secondary" size="large">Large Secondary Button</Button>
<Button type="outline">Outline Button</Button>
```

### 5. CodeBlockWrapper Component
Use for enhanced code block presentation with optional titles.

```md
import CodeBlockWrapper from '@site/src/components/CodeBlockWrapper';

<CodeBlockWrapper title="Example Code">
```js
console.log('Hello, world!');
```
</CodeBlockWrapper>
```

## Typography Guidelines

- **H1**: Use for main page titles (automatically applied to main content)
- **H2**: Use for major section headings (creates visual separation)
- **H3**: Use for subsection headings
- **H4-H6**: Use for further subsections as needed

## Color Palette

- **Primary**: Green (`#2e7d32`) - Used for main actions and links
- **Secondary**: Blue-gray (`#607d8b`) - Used for secondary elements
- **Info**: Blue (`#0277bd`) - Used for information boxes
- **Success**: Green (`#388e3c`) - Used for success messages
- **Warning**: Orange (`#f57c00`) - Used for warnings
- **Danger**: Red (`#d32f2f`) - Used for error messages

## Responsive Design

All components are designed to be responsive and will automatically adjust for different screen sizes:
- Mobile: Optimized for screens up to 768px
- Tablet: Optimized for screens 768px to 996px
- Desktop: Optimized for screens above 996px

## Accessibility

All components follow accessibility best practices:
- Sufficient color contrast ratios
- Proper focus indicators
- Semantic HTML structure
- Screen reader compatibility

## Theme System

The site supports both light and dark modes:
- Automatic detection based on system preference
- Manual override available in the header
- Consistent styling across both themes