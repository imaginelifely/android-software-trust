"""Interactive Sand Physics & Particle Animation Component for Streamlit.

Provides a tactile, organic digital sand simulation where particles drift,
pile into dunes, and scatter when interacting with the user's cursor.
"""

def render_sand_hero(title: str = "Source-Aware Evidence Fusion", subtitle: str = "Multi-Source Security Risk Assessment") -> str:
    """Generate HTML/JS for an interactive sand particle hero banner."""
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: transparent;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        #sand-container {{
            position: relative;
            width: 100%;
            height: 200px;
            background: radial-gradient(circle at 50% 30%, #161b26 0%, #080b11 100%);
            border-radius: 12px;
            border: 1px solid rgba(230, 184, 125, 0.25);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(230, 184, 125, 0.2);
            overflow: hidden;
        }}
        canvas {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: block;
        }}
        .hero-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            pointer-events: none;
            z-index: 10;
            padding: 0 20px;
        }}
        .hero-badge {{
            background: rgba(230, 184, 125, 0.12);
            border: 1px solid rgba(230, 184, 125, 0.35);
            color: #E6B87D;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
            box-shadow: 0 0 15px rgba(230, 184, 125, 0.15);
        }}
        .hero-title {{
            color: #F8FAFC;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.8);
        }}
        .hero-sub {{
            color: #94A3B8;
            font-size: 12px;
            max-width: 650px;
            line-height: 1.4;
        }}
        .hint-badge {{
            position: absolute;
            bottom: 8px;
            right: 12px;
            color: rgba(230, 184, 125, 0.6);
            font-size: 10px;
            letter-spacing: 0.05em;
            pointer-events: none;
            z-index: 11;
        }}
    </style>
    </head>
    <body>
    <div id="sand-container">
        <canvas id="sand-canvas"></canvas>
        <div class="hero-overlay">
            <div class="hero-badge">Research Paper Companion & Engine</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-sub">{subtitle}</p>
        </div>
        <div class="hint-badge">✨ Interactive Kinetic Sand Physics — Move cursor to scatter grains</div>
    </div>

    <script>
        const canvas = document.getElementById('sand-canvas');
        const ctx = canvas.getContext('2d');
        let width, height;

        function resize() {{
            width = canvas.width = canvas.offsetWidth;
            height = canvas.height = canvas.offsetHeight;
        }}
        window.addEventListener('resize', resize);
        resize();

        // Sand grain palette
        const colors = [
            '#FDE68A', // Pale quartz
            '#E6B87D', // Golden desert sand
            '#D97706', // Amber grain
            '#C2410C', // Terracotta mineral
            '#2DD4BF', // Cyber teal spark
            '#38BDF8', // Luminous azure
            '#94A3B8'  // Obsidian basalt
        ];

        const GRAINS_COUNT = 380;
        const grains = [];

        class SandGrain {{
            constructor() {{
                this.reset(true);
            }}

            reset(initial = false) {{
                this.x = Math.random() * width;
                this.y = initial ? Math.random() * height : -10;
                this.size = Math.random() * 2.2 + 1.0;
                this.vx = (Math.random() - 0.5) * 0.6;
                this.vy = Math.random() * 0.9 + 0.4;
                this.color = colors[Math.floor(Math.random() * colors.length)];
                this.alpha = Math.random() * 0.6 + 0.35;
                this.friction = 0.98;
                this.gravity = 0.02;
            }}

            update(mouseX, mouseY, isHovering) {{
                // Gentle gravity & drift
                this.vy += this.gravity;
                this.vx *= this.friction;
                this.vy *= this.friction;

                // Ambient desert breeze
                this.x += this.vx + Math.sin(this.y * 0.02 + Date.now() * 0.001) * 0.35;
                this.y += this.vy;

                // Mouse interaction - sand scatter
                if (isHovering && mouseX !== null) {{
                    const dx = this.x - mouseX;
                    const dy = this.y - mouseY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    const forceRadius = 90;

                    if (dist < forceRadius && dist > 0) {{
                        const force = (1 - dist / forceRadius) * 2.5;
                        this.vx += (dx / dist) * force;
                        this.vy += (dy / dist) * force;
                    }}
                }}

                // Bottom settling / bounce
                if (this.y > height - 6) {{
                    this.y = height - 6 - Math.random() * 3;
                    this.vy = -this.vy * 0.2;
                    this.vx += (Math.random() - 0.5) * 0.5;
                    
                    // Reset when settled too long
                    if (Math.abs(this.vy) < 0.05 && Math.random() < 0.015) {{
                        this.reset();
                    }}
                }}

                if (this.x < 0) this.x = width;
                if (this.x > width) this.x = 0;
            }}

            draw() {{
                ctx.save();
                ctx.globalAlpha = this.alpha;
                ctx.fillStyle = this.color;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }}
        }}

        for (let i = 0; i < GRAINS_COUNT; i++) {{
            grains.push(new SandGrain());
        }}

        let mouseX = null;
        let mouseY = null;
        let isHovering = false;

        const container = document.getElementById('sand-container');
        container.addEventListener('mousemove', (e) => {{
            const rect = canvas.getBoundingClientRect();
            mouseX = e.clientX - rect.left;
            mouseY = e.clientY - rect.top;
            isHovering = true;
        }});
        container.addEventListener('mouseleave', () => {{
            mouseX = null;
            mouseY = null;
            isHovering = false;
        }});

        function animate() {{
            ctx.clearRect(0, 0, width, height);

            // Draw subtle sand dune base contour
            ctx.save();
            ctx.fillStyle = 'rgba(230, 184, 125, 0.04)';
            ctx.beginPath();
            ctx.moveTo(0, height);
            for (let x = 0; x <= width; x += 30) {{
                const y = height - 14 + Math.sin(x * 0.015 + Date.now() * 0.0005) * 6;
                ctx.lineTo(x, y);
            }}
            ctx.lineTo(width, height);
            ctx.closePath();
            ctx.fill();
            ctx.restore();

            for (let g of grains) {{
                g.update(mouseX, mouseY, isHovering);
                g.draw();
            }}
            requestAnimationFrame(animate);
        }}
        animate();
    </script>
    </body>
    </html>
    """
    return html_code
