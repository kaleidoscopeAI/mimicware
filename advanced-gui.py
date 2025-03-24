    """_summary_

    Raises:
        ValueError: _description_
        ValueError: _description_
        FileNotFoundError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        FileNotFoundError: _description_

    Returns:
        _type_: _description_
    """#!/usr/bin/env python3
"""
Advanced Drag-and-Drop Interface for the System Builder
Implements a customizable canvas with node-based workflow editing
"""

import os
import sys
import json
import uuid
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import numpy as np
import threading
import queue
import re
import time
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable

# Theme colors
THEME = {
    'bg': '#2E3440',
    'fg': '#ECEFF4',
    'accent': '#88C0D0',
    'success': '#A3BE8C',
    'warning': '#EBCB8B',
    'error': '#BF616A',
    'node': '#4C566A',
    'node_selected': '#5E81AC',
    'connection': '#D8DEE9',
    'canvas_bg': '#3B4252',
    'panel_bg': '#2E3440',
    'panel_fg': '#ECEFF4',
    'font': ('Segoe UI', 10),
    'font_bold': ('Segoe UI', 10, 'bold'),
    'font_header': ('Segoe UI', 16, 'bold'),
    'font_subheader': ('Segoe UI', 12, 'bold')
}

# ======================================================================
# Custom Styling and Widgets
# ======================================================================

def setup_styles():
    """Setup ttk styles for the application"""
    style = ttk.Style()
    
    # Create custom theme
    style.theme_create("SystemBuilder", parent="clam", settings={
        "TFrame": {
            "configure": {
                "background": THEME['bg']
            }
        },
        "TLabel": {
            "configure": {
                "background": THEME['bg'],
                "foreground": THEME['fg'],
                "font": THEME['font']
            }
        },
        "TButton": {
            "configure": {
                "background": THEME['accent'],
                "foreground": THEME['bg'],
                "font": THEME['font_bold'],
                "padding": (10, 5)
            },
            "map": {
                "background": [("active", THEME['accent']), ("pressed", THEME['bg'])],
                "foreground": [("active", THEME['bg']), ("pressed", THEME['fg'])]
            }
        },
        "TCheckbutton": {
            "configure": {
                "background": THEME['bg'],
                "foreground": THEME['fg'],
                "font": THEME['font']
            }
        },
        "TEntry": {
            "configure": {
                "fieldbackground": THEME['panel_bg'],
                "foreground": THEME['fg'],
                "insertcolor": THEME['fg'],
                "font": THEME['font']
            }
        },
        "TCombobox": {
            "configure": {
                "fieldbackground": THEME['panel_bg'],
                "background": THEME['accent'],
                "foreground": THEME['fg'],
                "font": THEME['font']
            }
        },
        "TNotebook": {
            "configure": {
                "background": THEME['bg'],
                "tabmargins": [2, 5, 2, 0]
            }
        },
        "TNotebook.Tab": {
            "configure": {
                "background": THEME['node'],
                "foreground": THEME['fg'],
                "padding": [10, 2],
                "font": THEME['font']
            },
            "map": {
                "background": [("selected", THEME['accent'])],
                "foreground": [("selected", THEME['bg'])],
                "expand": [("selected", [1, 1, 1, 0])]
            }
        }
    })
    
    # Use the new theme
    style.theme_use("SystemBuilder")

class ThemedFrame(ttk.Frame):
    """A themed frame widget"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(style="TFrame")

class ThemedLabel(ttk.Label):
    """A themed label widget"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(style="TLabel")

class ThemedButton(ttk.Button):
    """A themed button widget"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(style="TButton")

class RoundedButton(tk.Canvas):
    """A rounded button with hover effects"""
    def __init__(self, master=None, text="", command=None, width=100, height=30, 
                corner_radius=10, bg=THEME['accent'], fg=THEME['bg'], 
                hover_bg=THEME['node_selected'], hover_fg=THEME['fg'], **kwargs):
        super().__init__(master, width=width, height=height, 
                        bg=THEME['bg'], highlightthickness=0, **kwargs)
        
        self.text = text
        self.command = command
        self.width = width
        self.height = height
        self.corner_radius = corner_radius
        self.bg = bg
        self.fg = fg
        self.hover_bg = hover_bg
        self.hover_fg = hover_fg
        self.hovered = False
        
        # Draw the initial button
        self.draw_button()
        
        # Bind events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)
    
    def draw_button(self):
        """Draw the button"""
        self.delete("all")
        
        # Choose colors based on hover state
        bg_color = self.hover_bg if self.hovered else self.bg
        fg_color = self.hover_fg if self.hovered else self.fg
        
        # Create rounded rectangle
        self.create_rounded_rect(0, 0, self.width, self.height, self.corner_radius, fill=bg_color)
        
        # Add text
        self.create_text(self.width/2, self.height/2, text=self.text, fill=fg_color, 
                       font=THEME['font_bold'])
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Create a rounded rectangle"""
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def on_enter(self, event):
        """Handle mouse enter event"""
        self.hovered = True
        self.draw_button()
    
    def on_leave(self, event):
        """Handle mouse leave event"""
        self.hovered = False
        self.draw_button()
    
    def on_click(self, event):
        """Handle mouse click event"""
        if self.command:
            self.command()

class ScrollableFrame(ttk.Frame):
    """A scrollable frame"""
    def __init__(self, master=None, **kwargs):
        # Create a frame for the canvas and scrollbar
        container = ttk.Frame(master)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Create the canvas and scrollbar
        self.canvas = tk.Canvas(container, bg=THEME['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        
        # Configure the canvas
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create the scrollable frame inside the canvas
        super().__init__(self.canvas, **kwargs)
        
        # Add the frame to the canvas
        self.window = self.canvas.create_window((0, 0), window=self, anchor=tk.NW)
        
        # Bind events
        self.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", self.on_mousewheel)
        self.canvas.bind_all("<Button-5>", self.on_mousewheel)
    
    def on_frame_configure(self, event):
        """Update the canvas scrollregion"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """Resize the inner frame to fill the canvas"""
        width = event.width
        self.canvas.itemconfig(self.window, width=width)
    
    def on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")

class ToggleButton(tk.Canvas):
    """A toggle button with an on/off state"""
    def __init__(self, master=None, width=60, height=30, command=None, **kwargs):
        super().__init__(master, width=width, height=height, bg=THEME['bg'], 
                        highlightthickness=0, **kwargs)
        
        self.width = width
        self.height = height
        self.command = command
        self.state = False
        
        # Draw the initial button
        self.draw_button()
        
        # Bind events
        self.bind("<Button-1>", self.toggle)
    
    def draw_button(self):
        """Draw the toggle button"""
        self.delete("all")
        
        # Choose colors based on state
        bg_color = THEME['success'] if self.state else THEME['error']
        
        # Create track
        track_width = self.width - 4
        track_height = self.height // 2
        track_y = (self.height - track_height) // 2
        
        self.create_rounded_rect(2, track_y, track_width+2, track_y+track_height, 
                               track_height//2, fill=bg_color, outline="", width=0)
        
        # Create thumb
        thumb_size = track_height - 4
        thumb_y = track_y + 2
        thumb_x = track_width - thumb_size if self.state else 4
        
        self.create_oval(thumb_x, thumb_y, thumb_x+thumb_size, thumb_y+thumb_size, 
                       fill=THEME['fg'], outline="", width=0)
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Create a rounded rectangle"""
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, **kwargs, smooth=True)
    
    def toggle(self, event=None):
        """Toggle the button state"""
        self.state = not self.state
        self.draw_button()
        
        if self.command:
            self.command(self.state)

class ColorPicker(ttk.Frame):
    """A color picker widget"""
    def __init__(self, master=None, color=THEME['accent'], command=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.color = color
        self.command = command
        
        # Create a color preview button
        self.preview = tk.Canvas(self, width=30, height=20, bg=self.color, 
                               highlightthickness=1, highlightbackground=THEME['fg'])
        self.preview.pack(side=tk.LEFT, padx=5)
        
        # Create a button to open the color chooser
        self.button = ThemedButton(self, text="Choose", command=self.choose_color)
        self.button.pack(side=tk.LEFT, padx=5)
        
        # Bind click on preview
        self.preview.bind("<Button-1>", lambda e: self.choose_color())
    
    def choose_color(self):
        """Open the color chooser dialog"""
        color = colorchooser.askcolor(initialcolor=self.color)
        if color[1]:
            self.color = color[1]
            self.preview.configure(bg=self.color)
            
            if self.command:
                self.command(self.color)

class PropertyEditor(ttk.Frame):
    """A property editor widget for node properties"""
    def __init__(self, master=None, properties=None, on_change=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.properties = properties or {}
        self.on_change = on_change
        self.editors = {}
        
        # Create a scrollable frame for the properties
        self.scrollable_frame = ScrollableFrame(self)
        self.scrollable_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add editors for each property
        self.update_editors()
    
    def update_editors(self):
        """Update property editors based on current properties"""
        # Clear existing editors
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.editors = {}
        
        # Create editors for each property
        for key, value in self.properties.items():
            self.add_property_editor(key, value)
    
    def add_property_editor(self, key, value):
        """Add an editor for a property"""
        # Create a frame for this property
        frame = ttk.Frame(self.scrollable_frame)
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a label for the property name
        label = ThemedLabel(frame, text=key, width=15, anchor=tk.W)
        label.pack(side=tk.LEFT, padx=5)
        
        # Create an appropriate editor based on the value type
        if isinstance(value, bool):
            # Boolean value - use a toggle button
            editor = ToggleButton(frame, width=60, height=30, 
                                command=lambda s, k=key: self.property_changed(k, s))
            editor.state = value
            editor.draw_button()
            editor.pack(side=tk.LEFT, padx=5)
            
        elif isinstance(value, (int, float)):
            # Numeric value - use a spinbox
            var = tk.StringVar(value=str(value))
            editor = ttk.Spinbox(frame, from_=0, to=1000, textvariable=var, width=10)
            editor.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Bind change event
            var.trace_add("write", lambda *args, k=key, v=var: self.property_changed(k, 
                                                                                  self.parse_number(v.get())))
            
        elif isinstance(value, str) and (value.startswith('#') or value in list(THEME.values())):
            # Color value - use a color picker
            editor = ColorPicker(frame, color=value, 
                               command=lambda c, k=key: self.property_changed(k, c))
            editor.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
        elif isinstance(value, list):
            # List value - use a dropdown if short, or a listbox if long
            if len(value) <= 10 and all(isinstance(item, str) for item in value):
                var = tk.StringVar(value=value[0] if value else "")
                editor = ttk.Combobox(frame, values=value, textvariable=var, width=20)
                editor.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                
                # Bind change event
                var.trace_add("write", lambda *args, k=key, v=var: self.property_changed(k, v.get()))
                
            else:
                # Create a more complex editor for longer lists
                editor = tk.Listbox(frame, height=min(len(value), 5), bg=THEME['panel_bg'], 
                                  fg=THEME['fg'], selectbackground=THEME['accent'])
                scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=editor.yview)
                editor.configure(yscrollcommand=scrollbar.set)
                
                editor.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                # Add list items
                for item in value:
                    editor.insert(tk.END, str(item))
                
                # Bind change event
                editor.bind("<<ListboxSelect>>", 
                          lambda e, k=key, lb=editor: self.property_changed(k, lb.get(lb.curselection()[0]) 
                                                                         if lb.curselection() else None))
        else:
            # Default to a text entry
            var = tk.StringVar(value=str(value))
            editor = ttk.Entry(frame, textvariable=var)
            editor.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Bind change event
            var.trace_add("write", lambda *args, k=key, v=var: self.property_changed(k, v.get()))
        
        # Store the editor
        self.editors[key] = editor
    
    def parse_number(self, value):
        """Parse a string as a number (int or float)"""
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return 0
    
    def property_changed(self, key, value):
        """Handle property value change"""
        self.properties[key] = value
        
        if self.on_change:
            self.on_change(key, value)
    
    def get_properties(self):
        """Get the current properties"""
        return self.properties
    
    def set_properties(self, properties):
        """Set new properties"""
        self.properties = properties or {}
        self.update_editors()

# ======================================================================
# Canvas and Node Classes
# ======================================================================

class CanvasNode:
    """A node for the workflow canvas"""
    def __init__(self, canvas, x, y, node_id, title="Node", node_type="script", 
               properties=None, width=150, height=100, color=None):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.id = node_id
        self.title = title
        self.node_type = node_type
        self.properties = properties or {}
        self.width = width
        self.height = height
        self.color = color or THEME['node']
        
        # Node state
        self.selected = False
        self.input_ports = []
        self.output_ports = []
        self.connections = []
        
        # Create the node on the canvas
        self.create()
    
    def create(self):
        """Create the node on the canvas"""
        # Create node group
        self.group_id = self.canvas.create_rectangle(
            self.x, self.y, self.x + self.width, self.y + self.height,
            fill=self.color, outline=THEME['fg'], width=2,
            tags=("node", f"node:{self.id}")
        )
        
        # Create title bar
        title_height = 25
        self.title_id = self.canvas.create_rectangle(
            self.x, self.y, self.x + self.width, self.y + title_height,
            fill=self.get_title_color(), outline="",
            tags=("node_title", f"node:{self.id}")
        )
        
        # Create title text
        self.title_text_id = self.canvas.create_text(
            self.x + self.width/2, self.y + title_height/2,
            text=self.title, fill=THEME['fg'], font=THEME['font_bold'],
            tags=("node_text", f"node:{self.id}")
        )
        
        # Create node type indicator
        type_label = self.node_type.capitalize()
        self.type_id = self.canvas.create_text(
            self.x + self.width/2, self.y + title_height + 15,
            text=type_label, fill=THEME['fg'], font=THEME['font'],
            tags=("node_type", f"node:{self.id}")
        )
        
        # Create ports
        self.create_ports()
        
        # Bind events
        self.canvas.tag_bind(f"node:{self.id}", "<Button-1>", self.on_select)
        self.canvas.tag_bind(f"node:{self.id}", "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(f"node:{self.id}", "<Double-Button-1>", self.on_double_click)
    
    def create_ports(self):
        """Create input and output ports for the node"""
        # Clear existing ports
        for port in self.input_ports + self.output_ports:
            self.canvas.delete(port['id'])
            self.canvas.delete(port['text_id'])
        
        self.input_ports = []
        self.output_ports = []
        
        # Create input ports based on node type
        input_count = 0
        if self.node_type in ('script', 'computation', 'condition'):
            input_count = 1
        elif self.node_type == 'merge':
            input_count = 3
        
        port_spacing = 30
        for i in range(input_count):
            port_x = self.x
            port_y = self.y + 40 + i * port_spacing
            
            port_id = self.canvas.create_oval(
                port_x - 5, port_y - 5, port_x + 5, port_y + 5,
                fill=THEME['accent'], outline=THEME['fg'],
                tags=("port", "input_port", f"node:{self.id}", f"port:{self.id}:{i}:input")
            )
            
            port_text_id = self.canvas.create_text(
                port_x + 10, port_y,
                text=f"In {i+1}", fill=THEME['fg'], font=THEME['font'],
                anchor=tk.W,
                tags=("port_text", f"node:{self.id}")
            )
            
            self.input_ports.append({
                'id': port_id,
                'text_id': port_text_id,
                'x': port_x,
                'y': port_y,
                'index': i,
                'type': 'input'
            })
            
            # Bind events
            self.canvas.tag_bind(port_id, "<Button-1>", 
                               lambda e, p=self.input_ports[-1]: self.on_port_click(e, p))
        
        # Create output ports
        output_count = 0
        if self.node_type in ('script', 'data', 'computation'):
            output_count = 1
        elif self.node_type == 'condition':
            output_count = 2  # True/False outputs
        
        for i in range(output_count):
            port_x = self.x + self.width
            port_y = self.y + 40 + i * port_spacing
            
            port_id = self.canvas.create_oval(
                port_x - 5, port_y - 5, port_x + 5, port_y + 5,
                fill=THEME['success'], outline=THEME['fg'],
                tags=("port", "output_port", f"node:{self.id}", f"port:{self.id}:{i}:output")
            )
            
            label = f"Out {i+1}"
            if self.node_type == 'condition' and i < 2:
                label = "True" if i == 0 else "False"
            
            port_text_id = self.canvas.create_text(
                port_x - 10, port_y,
                text=label, fill=THEME['fg'], font=THEME['font'],
                anchor=tk.E,
                tags=("port_text", f"node:{self.id}")
            )
            
            self.output_ports.append({
                'id': port_id,
                'text_id': port_text_id,
                'x': port_x,
                'y': port_y,
                'index': i,
                'type': 'output'
            })
            
            # Bind events
            self.canvas.tag_bind(port_id, "<Button-1>", 
                               lambda e, p=self.output_ports[-1]: self.on_port_click(e, p))
    
    def get_title_color(self):
        """Get color for the title bar based on node type"""
        if self.node_type == 'script':
            return THEME['accent']
        elif self.node_type == 'data':
            return THEME['success']
        elif self.node_type == 'computation':
            return "#9370DB"  # Purple
        elif self.node_type == 'condition':
            return THEME['warning']
        elif self.node_type == 'resource':
            return "#4682B4"  # Steel Blue
        elif self.node_type == 'merge':
            return "#CD853F"  # Peru
        else:
            return THEME['node']
    
    def on_select(self, event):
        """Handle node selection"""
        # Select this node
        self.canvas.workflow.select_node(self.id)
    
    def on_drag(self, event):
        """Handle node dragging"""
        if not self.selected:
            return
        
        # Calculate movement delta
        dx = event.x - self.canvas.last_mouse_x
        dy = event.y - self.canvas.last_mouse_y
        
        # Move the node and all its parts
        self.canvas.move(f"node:{self.id}", dx, dy)
        
        # Update coordinates
        self.x += dx
        self.y += dy
        
        # Update port positions
        for port in self.input_ports:
            port['x'] += dx
            port['y'] += dy
        
        for port in self.output_ports:
            port['x'] += dx
            port['y'] += dy
        
        # Update connections
        self.canvas.workflow.update_connections()
        
        # Update last mouse position
        self.canvas.last_mouse_x = event.x
        self.canvas.last_mouse_y = event.y
    
    def on_double_click(self, event):
        """Handle node double-click"""
        # Open the properties dialog
        self.canvas.workflow.edit_node_properties(self.id)
    
    def on_port_click(self, event, port):
        """Handle port click for connections"""
        # Start or end a connection
        self.canvas.workflow.handle_port_click(self.id, port)
    
    def set_selected(self, selected):
        """Set the selection state of the node"""
        self.selected = selected
        
        # Update appearance
        outline_width = 3 if selected else 2
        outline_color = THEME['node_selected'] if selected else THEME['fg']
        
        self.canvas.itemconfigure(self.group_id, width=outline_width, outline=outline_color)
    
    def update(self):
        """Update the node appearance based on properties"""
        # Update title
        self.canvas.itemconfigure(self.title_text_id, text=self.title)
        
        # Update colors
        self.canvas.itemconfigure(self.title_id, fill=self.get_title_color())
        
        # Update positions
        self.canvas.coords(self.group_id, 
                         self.x, self.y, self.x + self.width, self.y + self.height)
        
        title_height = 25
        self.canvas.coords(self.title_id, 
                         self.x, self.y, self.x + self.width, self.y + title_height)
        
        self.canvas.coords(self.title_text_id, 
                         self.x + self.width/2, self.y + title_height/2)
        
        self.canvas.coords(self.type_id, 
                         self.x + self.width/2, self.y + title_height + 15)
        
        # Recreate ports
        self.create_ports()

class Connection:
    """A connection between node ports"""
    def __init__(self, canvas, source_node_id, source_port, target_node_id, target_port, 
               connection_type="control"):
        self.canvas = canvas
        self.source_node_id = source_node_id
        self.source_port = source_port
        self.target_node_id = target_node_id
        self.target_port = target_port
        self.connection_type = connection_type
        
        # Create the connection line
        self.create()
    
    def create(self):
        """Create the connection line on the canvas"""
        # Get source and target positions
        source_node = self.canvas.workflow.get_node(self.source_node_id)
        target_node = self.canvas.workflow.get_node(self.target_node_id)
        
        if not source_node or not target_node:
            return
        
        source_x = source_node.x + source_node.width
        source_y = source_node.y + 40 + self.source_port['index'] * 30
        
        target_x = target_node.x
        target_y = target_node.y + 40 + self.target_port['index'] * 30
        
        # Calculate control points for curve
        ctrl_dist = 50  # Distance of control points from endpoints
        
        # Create a curved line (bezier curve)
        curve_points = [
            source_x, source_y,
            source_x + ctrl_dist, source_y,
            target_x - ctrl_dist, target_y,
            target_x, target_y
        ]
        
        # Choose color based on connection type
        color = THEME['connection']
        if self.connection_type == 'data':
            color = THEME['success']
        elif self.connection_type == 'control':
            color = THEME['accent']
        
        # Create the line
        self.line_id = self.canvas.create_line(
            curve_points, smooth=True, width=2, fill=color,
            arrow=tk.LAST, arrowshape=(10, 12, 5),
            tags=("connection", f"connection:{self.source_node_id}:{self.target_node_id}")
        )
        
        # Add a delete button at the middle of the connection
        mid_x = (source_x + target_x) / 2
        mid_y = (source_y + target_y) / 2
        
        self.delete_btn = self.canvas.create_oval(
            mid_x-5, mid_y-5, mid_x+5, mid_y+5,
            fill=THEME['error'], outline=THEME['fg'],
            tags=("connection_delete", f"connection_delete:{self.source_node_id}:{self.target_node_id}")
        )
        
        # Bind events to the delete button
        self.canvas.tag_bind(self.delete_btn, "<Button-1>", self.on_delete)
    
    def update(self):
        """Update the connection position"""
        source_node = self.canvas.workflow.get_node(self.source_node_id)
        target_node = self.canvas.workflow.get_node(self.target_node_id)
        
        if not source_node or not target_node:
            return
        
        # Find the correct port coordinates
        source_port = None
        for port in source_node.output_ports:
            if port['index'] == self.source_port['index']:
                source_port = port
                break
        
        target_port = None
        for port in target_node.input_ports:
            if port['index'] == self.target_port['index']:
                target_port = port
                break
        
        if not source_port or not target_port:
            return
        
        source_x = source_port['x']
        source_y = source_port['y']
        
        target_x = target_port['x']
        target_y = target_port['y']
        
        # Calculate control points for curve
        ctrl_dist = 50  # Distance of control points from endpoints
        
        # Update the curve points
        curve_points = [
            source_x, source_y,
            source_x + ctrl_dist, source_y,
            target_x - ctrl_dist, target_y,
            target_x, target_y
        ]
        
        self.canvas.coords(self.line_id, curve_points)
        
        # Update delete button position
        mid_x = (source_x + target_x) / 2
        mid_y = (source_y + target_y) / 2
        
        self.canvas.coords(self.delete_btn, mid_x-5, mid_y-5, mid_x+5, mid_y+5)
    
    def on_delete(self, event):
        """Handle connection deletion"""
        self.canvas.workflow.delete_connection(self.source_node_id, self.target_node_id)

class WorkflowCanvas(tk.Canvas):
    """Canvas for building workflow diagrams"""
    def __init__(self, master=None, **kwargs):
        super().__init__(master, bg=THEME['canvas_bg'], highlightthickness=0, **kwargs)
        
        # Initialize attributes
        self.workflow = None
        self.nodes = {}
        self.connections = []
        self.selected_node_id = None
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Connection state
        self.connecting = False
        self.connection_source = None
        self.temp_connection_line = None
        
        # Pan and zoom state
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.scale = 1.0
        
        # Bind canvas events
        self.bind("<Button-1>", self.on_canvas_click)
        self.bind("<B1-Motion>", self.on_canvas_drag)
        self.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.bind("<Button-3>", self.on_right_click)
        self.bind("<Button-2>", self.on_middle_click)
        self.bind("<B2-Motion>", self.on_middle_drag)
        self.bind("<MouseWheel>", self.on_mousewheel)
        self.bind("<Button-4>", self.on_mousewheel)
        self.bind("<Button-5>", self.on_mousewheel)
        self.bind("<Configure>", self.on_resize)
        
        # Create a grid background
        self.grid_spacing = 20
        self.draw_grid()
    
    def set_workflow(self, workflow):
        """Set the workflow controller"""
        self.workflow = workflow
    
    def draw_grid(self):
        """Draw a grid on the canvas"""
        self.delete("grid")
        
        width = self.winfo_width()
        height = self.winfo_height()
        
        if width <= 1 or height <= 1:
            # Canvas not properly sized yet
            return
        
        # Draw vertical lines
        for x in range(0, width, self.grid_spacing):
            self.create_line(x, 0, x, height, fill="#3F444F", width=1, tags="grid")
        
        # Draw horizontal lines
        for y in range(0, height, self.grid_spacing):
            self.create_line(0, y, width, y, fill="#3F444F", width=1, tags="grid")
    
    def on_resize(self, event):
        """Handle canvas resize"""
        self.draw_grid()
    
    def on_canvas_click(self, event):
        """Handle canvas click"""
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        
        # Deselect all nodes if clicked on empty canvas
        if not self.find_withtag(tk.CURRENT) or "grid" in self.gettags(tk.CURRENT):
            if self.connecting:
                # Cancel connection
                self.cancel_connection()
            else:
                # Deselect node
                self.workflow.select_node(None)
    
    def on_canvas_drag(self, event):
        """Handle canvas drag"""
        if self.connecting:
            # Update temporary connection line
            self.update_temp_connection(event.x, event.y)
        
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
    
    def on_canvas_release(self, event):
        """Handle canvas release"""
        pass
    
    def on_right_click(self, event):
        """Handle right click for context menu"""
        # Check if clicked on a node
        node_id = None
        for item_id in self.find_closest(event.x, event.y):
            tags = self.gettags(item_id)
            for tag in tags:
                if tag.startswith("node:"):
                    node_id = tag.split(":")[1]
                    break
            if node_id:
                break
        
        # Create the context menu
        menu = tk.Menu(self, tearoff=0, bg=THEME['panel_bg'], fg=THEME['fg'],
                     activebackground=THEME['accent'], activeforeground=THEME['bg'])
        
        if node_id:
            # Node context menu
            menu.add_command(label="Edit Properties", 
                           command=lambda: self.workflow.edit_node_properties(node_id))
            menu.add_command(label="Delete Node", 
                           command=lambda: self.workflow.delete_node(node_id))
            menu.add_separator()
            
            # Add node-specific actions based on type
            node = self.workflow.get_node(node_id)
            if node and node.node_type == 'script':
                menu.add_command(label="Open Script", 
                               command=lambda: self.workflow.open_script(node_id))
            
            elif node and node.node_type == 'condition':
                menu.add_command(label="Edit Condition", 
                               command=lambda: self.workflow.edit_condition(node_id))
        
        else:
            # Canvas context menu
            menu.add_command(label="Add Script Node", 
                           command=lambda: self.workflow.add_node("script", event.x, event.y))
            menu.add_command(label="Add Data Node", 
                           command=lambda: self.workflow.add_node("data", event.x, event.y))
            menu.add_command(label="Add Computation Node", 
                           command=lambda: self.workflow.add_node("computation", event.x, event.y))
            menu.add_command(label="Add Condition Node", 
                           command=lambda: self.workflow.add_node("condition", event.x, event.y))
            menu.add_command(label="Add Resource Node", 
                           command=lambda: self.workflow.add_node("resource", event.x, event.y))
            menu.add_command(label="Add Merge Node", 
                           command=lambda: self.workflow.add_node("merge", event.x, event.y))
            menu.add_separator()
            menu.add_command(label="Select All", command=self.workflow.select_all)
            menu.add_command(label="Clear Canvas", command=self.workflow.clear_canvas)
            menu.add_separator()
            menu.add_command(label="Zoom In", command=lambda: self.zoom(1.1, event.x, event.y))
            menu.add_command(label="Zoom Out", command=lambda: self.zoom(0.9, event.x, event.y))
            menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        
        menu.tk_popup(event.x_root, event.y_root)
    
    def on_middle_click(self, event):
        """Handle middle click for panning"""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.config(cursor="fleur")
    
    def on_middle_drag(self, event):
        """Handle middle drag for panning"""
        if not self.panning:
            return
        
        # Calculate movement delta
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        # Pan the canvas
        self.scan_dragto(event.x, event.y, gain=1)
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def on_mousewheel(self, event):
        """Handle mousewheel for zooming"""
        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            self.zoom(1.1, event.x, event.y)
        elif event.num == 5 or event.delta < 0:
            self.zoom(0.9, event.x, event.y)
    
    def zoom(self, factor, x, y):
        """Zoom the canvas"""
        # Limit zoom level
        new_scale = self.scale * factor
        if new_scale < 0.2 or new_scale > 5.0:
            return
        
        # Apply zoom
        self.scale = new_scale
        self.scale_canvas(factor, x, y)
    
    def scale_canvas(self, factor, x, y):
        """Scale all canvas items"""
        # Scale all items
        self.scale("all", x, y, factor, factor)
        
        # Redraw grid at new scale
        self.draw_grid()
        
        # Update node positions for internal tracking
        for node_id, node in self.nodes.items():
            # Calculate new position based on zoom
            node.x = self.canvasx(self.canvasx(node.x) - x, 0) * factor + x
            node.y = self.canvasy(self.canvasy(node.y) - y, 0) * factor + y
            
            # Update port positions
            for port in node.input_ports:
                port['x'] = self.canvasx(self.canvasx(port['x']) - x, 0) * factor + x
                port['y'] = self.canvasy(self.canvasy(port['y']) - y, 0) * factor + y
            
            for port in node.output_ports:
                port['x'] = self.canvasx(self.canvasx(port['x']) - x, 0) * factor + x
                port['y'] = self.canvasy(self.canvasy(port['y']) - y, 0) * factor + y
    
    def reset_zoom(self):
        """Reset zoom to original scale"""
        # Calculate zoom factor to reset
        factor = 1.0 / self.scale
        
        # Get canvas center
        center_x = self.winfo_width() / 2
        center_y = self.winfo_height() / 2
        
        # Reset scale
        self.scale = 1.0
        self.scale_canvas(factor, center_x, center_y)
    
    def add_node(self, node_id, title, node_type, x, y, properties=None):
        """Add a node to the canvas"""
        node = CanvasNode(self, x, y, node_id, title, node_type, properties)
        self.nodes[node_id] = node
        return node
    
    def delete_node(self, node_id):
        """Delete a node from the canvas"""
        if node_id not in self.nodes:
            return
        
        # Delete all canvas items for this node
        self.delete(f"node:{node_id}")
        
        # Delete all connections to/from this node
        self.delete_connections_for_node(node_id)
        
        # Remove from nodes dict
        del self.nodes[node_id]
        
        # Clear selection if this was the selected node
        if self.selected_node_id == node_id:
            self.selected_node_id = None
    
    def select_node(self, node_id):
        """Select a node on the canvas"""
        # Deselect current selection
        if self.selected_node_id and self.selected_node_id in self.nodes:
            self.nodes[self.selected_node_id].set_selected(False)
        
        # Set new selection
        self.selected_node_id = node_id
        
        if node_id and node_id in self.nodes:
            self.nodes[node_id].set_selected(True)
    
    def start_connection(self, node_id, port):
        """Start creating a connection from a port"""
        # Set connection state
        self.connecting = True
        self.connection_source = {
            'node_id': node_id,
            'port': port
        }
        
        # Create temporary connection line
        source_x = port['x']
        source_y = port['y']
        
        self.temp_connection_line = self.create_line(
            source_x, source_y, source_x, source_y,
            width=2, fill=THEME['accent'], dash=(5, 3),
            arrow=tk.LAST, arrowshape=(10, 12, 5),
            tags="temp_connection"
        )
    
    def update_temp_connection(self, x, y):
        """Update the temporary connection line"""
        if not self.connecting or not self.temp_connection_line:
            return
        
        # Get source port position
        source_node = self.nodes.get(self.connection_source['node_id'])
        if not source_node:
            return
        
        port = self.connection_source['port']
        source_x = port['x']
        source_y = port['y']
        
        # Update the line
        self.coords(self.temp_connection_line, source_x, source_y, x, y)
    
    def finish_connection(self, target_node_id, target_port):
        """Finish creating a connection to a target port"""
        if not self.connecting or not self.connection_source:
            return
        
        source_node_id = self.connection_source['node_id']
        source_port = self.connection_source['port']
        
        # Create the connection
        self.workflow.add_connection(source_node_id, source_port, target_node_id, target_port)
        
        # Clean up temporary connection
        self.delete(self.temp_connection_line)
        self.connecting = False
        self.connection_source = None
        self.temp_connection_line = None
    
    def cancel_connection(self):
        """Cancel the current connection"""
        if not self.connecting:
            return
        
        # Clean up temporary connection
        if self.temp_connection_line:
            self.delete(self.temp_connection_line)
        
        self.connecting = False
        self.connection_source = None
        self.temp_connection_line = None
    
    def add_connection(self, source_node_id, source_port, target_node_id, target_port, connection_type="control"):
        """Add a connection between nodes"""
        # Create the connection
        connection = Connection(self, source_node_id, source_port, target_node_id, target_port, connection_type)
        self.connections.append(connection)
        
        return connection
    
    def delete_connection(self, source_node_id, target_node_id):
        """Delete a connection between nodes"""
        # Find the connection
        connection_to_delete = None
        for connection in self.connections:
            if connection.source_node_id == source_node_id and connection.target_node_id == target_node_id:
                connection_to_delete = connection
                break
        
        if not connection_to_delete:
            return
        
        # Delete the connection line and button
        self.delete(connection_to_delete.line_id)
        self.delete(connection_to_delete.delete_btn)
        
        # Remove from connections list
        self.connections.remove(connection_to_delete)
    
    def delete_connections_for_node(self, node_id):
        """Delete all connections for a node"""
        connections_to_delete = []
        
        for connection in self.connections:
            if connection.source_node_id == node_id or connection.target_node_id == node_id:
                connections_to_delete.append(connection)
        
        for connection in connections_to_delete:
            # Delete the connection line and button
            self.delete(connection.line_id)
            self.delete(connection.delete_btn)
            
            # Remove from connections list
            self.connections.remove(connection)
    
    def update_connections(self):
        """Update all connections"""
        for connection in self.connections:
            connection.update()

# ======================================================================
# Workflow Controller
# ======================================================================

class WorkflowController:
    """Controller for the workflow canvas"""
    def __init__(self, canvas: WorkflowCanvas, properties_panel=None):
        self.canvas = canvas
        self.properties_panel = properties_panel
        self.canvas.set_workflow(self)
        
        # Node and connection data
        self.nodes = {}
        self.connections = []
        
        # Context data
        self.script_paths = {}
    
    def add_node(self, node_type, x, y, title=None, properties=None):
        """Add a node to the workflow"""
        # Generate a unique ID
        node_id = str(uuid.uuid4())
        
        # Create a default title if none provided
        if not title:
            title = f"{node_type.capitalize()} {len(self.nodes) + 1}"
        
        # Create default properties based on node type
        if properties is None:
            properties = self.get_default_properties(node_type)
        
        # Add to canvas
        node = self.canvas.add_node(node_id, title, node_type, x, y, properties)
        
        # Store node data
        self.nodes[node_id] = {
            'id': node_id,
            'title': title,
            'type': node_type,
            'properties': properties
        }
        
        # Select the new node
        self.select_node(node_id)
        
        return node_id
    
    def get_default_properties(self, node_type):
        """Get default properties for a node type"""
        if node_type == 'script':
            return {
                'path': '',
                'script_type': 'python',
                'args': [],
                'env': {}
            }
        elif node_type == 'data':
            return {
                'data_type': 'static',
                'value': '',
                'file_path': ''
            }
        elif node_type == 'computation':
            return {
                'function': '',
                'params': {}
            }
        elif node_type == 'condition':
            return {
                'expression': 'True',
                'description': 'Default condition'
            }
        elif node_type == 'resource':
            return {
                'resource_type': 'file',
                'path': '',
                'operation': 'check'
            }
        elif node_type == 'merge':
            return {
                'strategy': 'dict',
                'description': 'Merge inputs'
            }
        
        return {}
    
    def delete_node(self, node_id):
        """Delete a node from the workflow"""
        if node_id not in self.nodes:
            return
        
        # Delete from canvas
        self.canvas.delete_node(node_id)
        
        # Remove from data
        del self.nodes[node_id]
        
        # Delete connections
        self.delete_connections_for_node(node_id)
        
        # Update properties panel
        if self.properties_panel:
            self.properties_panel.clear()
    
    def select_node(self, node_id):
        """Select a node in the workflow"""
        # Update canvas selection
        self.canvas.select_node(node_id)
        
        # Update properties panel
        if self.properties_panel:
            if node_id and node_id in self.nodes:
                node_data = self.nodes[node_id]
                self.properties_panel.set_node(node_data)
            else:
                self.properties_panel.clear()
    
    def get_node(self, node_id):
        """Get a canvas node by ID"""
        return self.canvas.nodes.get(node_id)
    
    def get_node_data(self, node_id):
        """Get node data by ID"""
        return self.nodes.get(node_id)
    
    def update_node(self, node_id, title=None, properties=None):
        """Update a node's properties"""
        if node_id not in self.nodes:
            return
        
        node_data = self.nodes[node_id]
        canvas_node = self.canvas.nodes.get(node_id)
        
        if not canvas_node:
            return
        
        # Update title if provided
        if title is not None:
            node_data['title'] = title
            canvas_node.title = title
        
        # Update properties if provided
        if properties is not None:
            node_data['properties'] = properties
            canvas_node.properties = properties
        
        # Update canvas node
        canvas_node.update()
        
        # Update connections
        self.canvas.update_connections()
    
    def handle_port_click(self, node_id, port):
        """Handle a click on a node port"""
        if not self.canvas.connecting:
            # Start connection
            if port['type'] == 'output':
                self.canvas.start_connection(node_id, port)
        else:
            # Finish connection
            source = self.canvas.connection_source
            
            # Check if this is a valid target
            if source['node_id'] == node_id:
                # Can't connect to self
                self.canvas.cancel_connection()
                return
            
            if port['type'] == 'output' or source['port']['type'] == 'input':
                # Invalid connection direction
                self.canvas.cancel_connection()
                return
            
            # Valid connection - finish it
            self.canvas.finish_connection(node_id, port)
    
    def add_connection(self, source_node_id, source_port, target_node_id, target_port, connection_type="control"):
        """Add a connection between nodes"""
        # Check for existing connection
        for conn in self.connections:
            if (conn['source_node'] == source_node_id and conn['target_node'] == target_node_id and
                conn['source_port_index'] == source_port['index'] and 
                conn['target_port_index'] == target_port['index']):
                # Connection already exists
                return None
        
        # Add to canvas
        connection = self.canvas.add_connection(
            source_node_id, source_port, target_node_id, target_port, connection_type
        )
        
        # Store connection data
        conn_data = {
            'source_node': source_node_id,
            'source_port_index': source_port['index'],
            'target_node': target_node_id,
            'target_port_index': target_port['index'],
            'type': connection_type
        }
        
        self.connections.append(conn_data)
        
        return conn_data
    
    def delete_connection(self, source_node_id, target_node_id):
        """Delete a connection between nodes"""
        # Delete from canvas
        self.canvas.delete_connection(source_node_id, target_node_id)
        
        # Remove from data
        self.connections = [conn for conn in self.connections 
                          if not (conn['source_node'] == source_node_id and 
                                 conn['target_node'] == target_node_id)]
    
    def delete_connections_for_node(self, node_id):
        """Delete all connections for a node"""
        # Delete from canvas
        self.canvas.delete_connections_for_node(node_id)
        
        # Remove from data
        self.connections = [conn for conn in self.connections 
                          if not (conn['source_node'] == node_id or 
                                 conn['target_node'] == node_id)]
    
    def update_connections(self):
        """Update all connections on the canvas"""
        self.canvas.update_connections()
    
    def edit_node_properties(self, node_id):
        """Edit a node's properties"""
        if not node_id or node_id not in self.nodes:
            return
        
        # Select the node
        self.select_node(node_id)
    
    def open_script(self, node_id):
        """Open a script in the default editor"""
        if not node_id or node_id not in self.nodes:
            return
        
        node_data = self.nodes[node_id]
        
        if node_data['type'] != 'script':
            return
        
        script_path = node_data['properties'].get('path', '')
        
        if not script_path or not os.path.exists(script_path):
            # Ask for the script path
            script_path = filedialog.askopenfilename(
                title="Select Script",
                filetypes=[
                    ("Python files", "*.py"),
                    ("Shell scripts", "*.sh"),
                    ("All files", "*.*")
                ]
            )
            
            if not script_path:
                return
            
            # Update node with the selected path
            node_data['properties']['path'] = script_path
            self.update_node(node_id, properties=node_data['properties'])
        
        # Open the script in the default editor
        try:
            if sys.platform.startswith('win'):
                os.startfile(script_path)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', script_path])
            else:
                subprocess.run(['xdg-open', script_path])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open script: {e}")
    
    def edit_condition(self, node_id):
        """Edit a condition node's expression"""
        if not node_id or node_id not in self.nodes:
            return
        
        node_data = self.nodes[node_id]
        
        if node_data['type'] != 'condition':
            return
        
        # Get the current expression
        expression = node_data['properties'].get('expression', 'True')
        
        # Create a dialog to edit the expression
        dialog = tk.Toplevel()
        dialog.title("Edit Condition")
        dialog.geometry("400x200")
        dialog.transient(self.canvas.master)
        dialog.grab_set()
        
        # Configure dialog style
        dialog.configure(bg=THEME['bg'])
        
        # Create dialog content
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Enter condition expression:").pack(anchor=tk.W, pady=(0, 5))
        
        text = tk.Text(frame, height=5, width=40, bg=THEME['panel_bg'], fg=THEME['fg'],
                     insertbackground=THEME['fg'])
        text.pack(fill=tk.BOTH, expand=True, pady=5)
        text.insert(tk.END, expression)
        
        # Create buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        def on_ok():
            new_expression = text.get("1.0", tk.END).strip()
            node_data['properties']['expression'] = new_expression
            self.update_node(node_id, properties=node_data['properties'])
            dialog.destroy()
        
        ttk.Button(button_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Focus the text widget
        text.focus_set()
    
    def select_all(self):
        """Select all nodes in the workflow"""
        # Currently only one node can be selected at a time
        pass
    
    def clear_canvas(self):
        """Clear all nodes and connections from the canvas"""
        # Ask for confirmation
        result = messagebox.askyesno("Confirm", "Clear all nodes and connections?")
        if not result:
            return
        
        # Delete all nodes
        for node_id in list(self.nodes.keys()):
            self.delete_node(node_id)
        
        # Clear connections
        self.connections = []
    
    def export_workflow(self):
        """Export the workflow to a JSON file"""
        # Create the workflow data
        workflow_data = {
            'nodes': self.nodes,
            'connections': self.connections,
            'metadata': {
                'created': time.strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0'
            }
        }
        
        # Ask for save location
        filepath = filedialog.asksaveasfilename(
            title="Export Workflow",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        # Save the file
        try:
            with open(filepath, 'w') as f:
                json.dump(workflow_data, f, indent=2)
            
            messagebox.showinfo("Export Successful", f"Workflow exported to {filepath}")
            return True
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export workflow: {e}")
            return False
    
    def import_workflow(self, filepath=None):
        """Import a workflow from a JSON file"""
        if not filepath:
            filepath = filedialog.askopenfilename(
                title="Import Workflow",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
        
        if not filepath:
            return False
        
        # Load the file
        try:
            with open(filepath, 'r') as f:
                workflow_data = json.load(f)
            
            # Clear current workflow
            self.clear_canvas()
            
            # Create nodes
            for node_id, node_data in workflow_data['nodes'].items():
                x = node_data.get('x', 100)
                y = node_data.get('y', 100)
                
                # Add the node
                self.add_node(
                    node_data['type'],
                    x, y,
                    node_data['title'],
                    node_data.get('properties', {})
                )
            
            # Create connections
            for conn in workflow_data['connections']:
                source_node_id = conn['source_node']
                target_node_id = conn['target_node']
                
                source_node = self.get_node(source_node_id)
                target_node = self.get_node(target_node_id)
                
                if source_node and target_node:
                    # Find the ports
                    source_port = None
                    for port in source_node.output_ports:
                        if port['index'] == conn['source_port_index']:
                            source_port = port
                            break
                    
                    target_port = None
                    for port in target_node.input_ports:
                        if port['index'] == conn['target_port_index']:
                            target_port = port
                            break
                    
                    if source_port and target_port:
                        # Add the connection
                        self.add_connection(
                            source_node_id, source_port,
                            target_node_id, target_port,
                            conn.get('type', 'control')
                        )
            
            messagebox.showinfo("Import Successful", f"Workflow imported from {filepath}")
            return True
        
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import workflow: {e}")
            return False
    
    def execute_workflow(self):
        """Execute the current workflow"""
        # Convert workflow to a format suitable for execution
        # This would typically be passed to a workflow engine
        
        # Check if there are any nodes
        if not self.nodes:
            messagebox.showinfo("Execution", "No nodes to execute")
            return
        
        # Find start nodes (nodes with no incoming connections)
        start_nodes = set(self.nodes.keys())
        
        for conn in self.connections:
            if conn['target_node'] in start_nodes:
                start_nodes.remove(conn['target_node'])
        
        if not start_nodes:
            messagebox.showwarning("Execution", "No start nodes found - workflow may have cycles")
            return
        
        # In a real implementation, this would execute the workflow
        # For now, just show a message
        messagebox.showinfo("Execution", 
                          f"Workflow execution started with {len(start_nodes)} start nodes")
        
        # Simulate execution by updating node status
        # In a real implementation, this would be updated by the workflow engine
        self.simulate_execution()
    
    def simulate_execution(self):
        """Simulate workflow execution for demonstration"""
        # Create a dialog to show execution progress
        dialog = tk.Toplevel()
        dialog.title("Workflow Execution")
        dialog.geometry("500x300")
        dialog.transient(self.canvas.master)
        
        # Configure dialog style
        dialog.configure(bg=THEME['bg'])
        
        # Create dialog content
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(frame, text="Execution Progress", font=THEME['font_header']).pack(pady=(0, 10))
        
        # Create a progress list
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        progress_list = tk.Listbox(list_frame, height=10, bg=THEME['panel_bg'], fg=THEME['fg'],
                                  selectbackground=THEME['accent'], yscrollcommand=scrollbar.set)
        progress_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=progress_list.yview)
        
        # Create a progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(frame, variable=progress_var, mode='determinate')
        progress_bar.pack(fill=tk.X, pady=10)
        
        # Create a status label
        status_var = tk.StringVar(value="Starting execution...")
        status_label = ttk.Label(frame, textvariable=status_var, font=THEME['font_bold'])
        status_label.pack(pady=5)
        
        # Create buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=dialog.destroy)
        cancel_button.pack(side=tk.RIGHT, padx=5)
        
        # Disable cancel button during execution
        cancel_button.config(state=tk.DISABLED)
        
        # Create a virtual graph for topological sorting
        try:
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes
            for node_id in self.nodes:
                G.add_node(node_id)
            
            # Add edges
            for conn in self.connections:
                G.add_edge(conn['source_node'], conn['target_node'])
            
            # Get execution order
            try:
                execution_order = list(nx.topological_sort(G))
            except nx.NetworkXUnfeasible:
                messagebox.showerror("Execution Error", "Workflow contains cycles and cannot be executed")
                dialog.destroy()
                return
            
            # Simulate execution
            total_nodes = len(execution_order)
            
            def simulate_node_execution(index):
                if index >= total_nodes:
                    # Execution complete
                    progress_var.set(100)
                    status_var.set("Execution complete")
                    progress_list.insert(tk.END, f"Workflow execution completed successfully")
                    progress_list.see(tk.END)
                    cancel_button.config(state=tk.NORMAL)
                    cancel_button.config(text="Close")
                    return
                
                # Get the node
                node_id = execution_order[index]
                node_data = self.nodes[node_id]
                
                # Update progress
                progress_var.set((index / total_nodes) * 100)
                status_var.set(f"Executing: {node_data['title']}")
                
                # Add to progress list
                progress_list.insert(tk.END, f"Executing: {node_data['title']} ({node_data['type']})")
                progress_list.see(tk.END)
                
                # Simulate execution time based on node type
                if node_data['type'] == 'script':
                    delay = 1500  # 1.5 seconds
                elif node_data['type'] == 'computation':
                    delay = 1000  # 1 second
                else:
                    delay = 500  # 0.5 seconds
                
                # Schedule next node execution
                dialog.after(delay, lambda: simulate_node_execution(index + 1))
            
            # Start execution simulation
            dialog.after(500, lambda: simulate_node_execution(0))
            
        except ImportError:
            messagebox.showerror("Execution Error", "NetworkX library is required for workflow execution")
            dialog.destroy()

# ======================================================================
# Property Panel
# ======================================================================

class PropertyPanel(ttk.Frame):
    """Panel for editing node properties"""
    def __init__(self, master=None, on_property_change=None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.on_property_change = on_property_change
        self.current_node = None
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        """Create UI widgets"""
        # Panel title
        self.title_label = ThemedLabel(self, text="Properties", font=THEME['font_header'])
        self.title_label.pack(pady=10)
        
        # Node info frame
        self.info_frame = ttk.Frame(self)
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Node title
        ttk.Label(self.info_frame, text="Title:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.title_var = tk.StringVar()
        self.title_entry = ttk.Entry(self.info_frame, textvariable=self.title_var)
        self.title_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=5)
        
        # Bind title change
        self.title_var.trace_add("write", self.on_title_change)
        
        # Node type
        ttk.Label(self.info_frame, text="Type:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        
        self.type_var = tk.StringVar()
        self.type_label = ttk.Label(self.info_frame, textvariable=self.type_var)
        self.type_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Configure grid
        self.info_frame.columnconfigure(1, weight=1)
        
        # Properties editor
        ttk.Label(self, text="Properties:", font=THEME['font_bold']).pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        self.property_editor = PropertyEditor(self, on_change=self.on_property_edited)
        self.property_editor.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Action buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.apply_button = ThemedButton(self.button_frame, text="Apply", command=self.apply_changes)
        self.apply_button.pack(side=tk.RIGHT, padx=5)
        
        self.reset_button = ThemedButton(self.button_frame, text="Reset", command=self.reset)
        self.reset_button.pack(side=tk.RIGHT, padx=5)
    
    def set_node(self, node_data):
        """Set the current node to edit"""
        self.current_node = node_data
        
        # Update UI
        self.title_var.set(node_data['title'])
        self.type_var.set(node_data['type'].capitalize())
        
        # Set properties
        self.property_editor.set_properties(node_data['properties'])
        
        # Enable buttons
        self.apply_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
    
    def clear(self):
        """Clear the property panel"""
        self.current_node = None
        
        # Clear UI
        self.title_var.set("")
        self.type_var.set("")
        
        # Clear properties
        self.property_editor.set_properties({})
        
        # Disable buttons
        self.apply_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
    
    def on_title_change(self, *args):
        """Handle title changes"""
        if not self.current_node:
            return
        
        new_title = self.title_var.get()
        if new_title != self.current_node['title']:
            self.current_node['title'] = new_title
            
            # Apply changes
            if self.on_property_change:
                self.on_property_change(self.current_node['id'], new_title)
    
    def on_property_edited(self, key, value):
        """Handle property edits"""
        if not self.current_node:
            return
        
        # Update the property
        self.current_node['properties'][key] = value
    
    def apply_changes(self):
        """Apply all property changes"""
        if not self.current_node or not self.on_property_change:
            return
        
        # Get updated properties
        properties = self.property_editor.get_properties()
        
        # Apply changes
        self.on_property_change(
            self.current_node['id'],
            self.title_var.get(),
            properties
        )
    
    def reset(self):
        """Reset property changes"""
        if not self.current_node:
            return
        
        # Reset to original values
        self.set_node(self.current_node)

# ======================================================================
# Script Template Manager
# ======================================================================

class ScriptTemplate:
    """A template for generating scripts"""
    def __init__(self, name, description, language, code_template):
        self.name = name
        self.description = description
        self.language = language
        self.code_template = code_template
        self.variables = self.extract_variables()
    
    def extract_variables(self):
        """Extract variables from the template"""
        variables = []
        pattern = r'\{\{([A-Za-z0-9_]+)\}\}'
        
        for match in re.finditer(pattern, self.code_template):
            var_name = match.group(1)
            if var_name not in variables:
                variables.append(var_name)
        
        return variables
    
    def render(self, variables):
        """Render the template with provided variables"""
        result = self.code_template
        
        for var_name, value in variables.items():
            result = result.replace(f'{{{{{var_name}}}}}', str(value))
        
        return result

class ScriptTemplateManager:
    """Manages script templates"""
    
    # Default templates
    DEFAULT_TEMPLATES = [
        ScriptTemplate(
            "Python Data Processing",
            "Script for processing data files",
            "python",
            """#!/usr/bin/env python3
# {{script_name}}.py
# {{description}}
# Created by: {{author}}
# Date: {{date}}

import os
import sys
import json
import csv
import argparse
from typing import Dict, List, Any

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="{{description}}")
    parser.add_argument('-i', '--input', required=True, help='Input file path')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv'], default='json', help='Output format')
    return parser.parse_args()

def read_data(file_path: str) -> List[Dict[str, Any]]:
    """Read data from the input file"""
    # Implement reading based on file extension
    _, ext = os.path.splitext(file_path)
    
    if ext.lower() == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    elif ext.lower() == '.csv':
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def process_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process the data"""
    result = []
    
    # Implement your data processing logic here
    for item in data:
        # Example: Add a new field
        processed_item = item.copy()
        processed_item['processed'] = True
        result.append(processed_item)
    
    return result

def write_data(data: List[Dict[str, Any]], file_path: str, format_type: str) -> None:
    """Write data to the output file"""
    if format_type == 'json':
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    elif format_type == 'csv':
        if not data:
            # No data to write
            return
            
        fieldnames = data[0].keys()
        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    else:
        raise ValueError(f"Unsupported output format: {format_type}")

def main():
    args = parse_arguments()
    
    try:
        # Read input data
        data = read_data(args.input)
        
        # Process data
        processed_data = process_data(data)
        
        # Write output
        write_data(processed_data, args.output, args.format)
        
        print(f"Processing complete: {len(data)} records processed")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        ),
        ScriptTemplate(
            "Bash Script",
            "Basic bash script template",
            "bash",
            """#!/bin/bash
# {{script_name}}.sh
# {{description}}
# Created by: {{author}}
# Date: {{date}}

# Exit on error
set -e

# Define variables
INPUT_DIR="{{input_dir}}"
OUTPUT_DIR="{{output_dir}}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-i|--input INPUT_DIR] [-o|--output OUTPUT_DIR]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-i|--input INPUT_DIR] [-o|--output OUTPUT_DIR]"
            exit 1
            ;;
    esac
done

# Check if directories exist
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process files
echo "Processing files from $INPUT_DIR to $OUTPUT_DIR"

# Your processing logic here
for file in "$INPUT_DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing $filename"
        
        # Example: Copy file to output directory
        cp "$file" "$OUTPUT_DIR/$filename"
    fi
done

echo "Processing complete!"
"""
        ),
        ScriptTemplate(
            "Node.js Script",
            "Basic Node.js script template",
            "javascript",
            """#!/usr/bin/env node
/**
 * {{script_name}}.js
 * {{description}}
 * Created by: {{author}}
 * Date: {{date}}
 */

const fs = require('fs');
const path = require('path');
const { promisify } = require('util');

// Promisify callbacks
const readFile = promisify(fs.readFile);
const writeFile = promisify(fs.writeFile);
const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);

// Parse command line arguments
const args = process.argv.slice(2);
const options = {};

for (let i = 0; i < args.length; i++) {
    if (args[i] === '-i' || args[i] === '--input') {
        options.inputPath = args[++i];
    } else if (args[i] === '-o' || args[i] === '--output') {
        options.outputPath = args[++i];
    } else if (args[i] === '-h' || args[i] === '--help') {
        console.log('Usage: node {{script_name}}.js [-i|--input INPUT_PATH] [-o|--output OUTPUT_PATH]');
        process.exit(0);
    }
}

// Validate options
if (!options.inputPath) {
    console.error('Input path is required. Use -i or --input to specify it.');
    process.exit(1);
}

if (!options.outputPath) {
    console.error('Output path is required. Use -o or --output to specify it.');
    process.exit(1);
}

/**
 * Main function
 */
async function main() {
    try {
        const inputStat = await stat(options.inputPath);
        
        if (inputStat.isFile()) {
            // Process a single file
            await processFile(options.inputPath, options.outputPath);
        } else if (inputStat.isDirectory()) {
            // Process a directory of files
            await processDirectory(options.inputPath, options.outputPath);
        } else {
            console.error(`Unsupported input: ${options.inputPath}`);
            process.exit(1);
        }
        
        console.log('Processing complete!');
    } catch (error) {
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
}

/**
 * Process a single file
 */
async function processFile(inputPath, outputPath) {
    console.log(`Processing file: ${inputPath}`);
    
    // Read the file
    const content = await readFile(inputPath, 'utf8');
    
    // Process the content
    const processedContent = processContent(content);
    
    // Write the result
    await writeFile(outputPath, processedContent, 'utf8');
    
    console.log(`Output written to: ${outputPath}`);
}

/**
 * Process a directory of files
 */
async function processDirectory(inputDir, outputDir) {
    console.log(`Processing directory: ${inputDir}`);
    
    // Ensure output directory exists
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Get list of files
    const files = await readdir(inputDir);
    
    // Process each file
    for (const file of files) {
        const inputPath = path.join(inputDir, file);
        const outputPath = path.join(outputDir, file);
        
        const fileStat = await stat(inputPath);
        
        if (fileStat.isFile()) {
            await processFile(inputPath, outputPath);
        }
    }
}

/**
 * Process content
 */
function processContent(content) {
    // Implement your processing logic here
    // This is just an example
    return content.toUpperCase();
}

// Run the main function
main();
"""
        ),
        ScriptTemplate(
            "Python Job Runner",
            "Script for running multiple jobs",
            "python",
            """#!/usr/bin/env python3
# {{script_name}}.py
# {{description}}
# Created by: {{author}}
# Date: {{date}}

import os
import sys
import time
import json
import argparse
import logging
import concurrent.futures
from typing import Dict, List, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("{{script_name}}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="{{description}}")
    parser.add_argument('-c', '--config', required=True, help='Configuration file path')
    parser.add_argument('-j', '--jobs', type=int, default=4, help='Number of parallel jobs')
    parser.add_argument('-o', '--output-dir', help='Output directory for job results')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', 
                      help='Logging level')
    return parser.parse_args()

def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    _, ext = os.path.splitext(config_path)
    
    if ext.lower() == '.json':
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {ext}")

def run_job(job_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single job"""
    job_id = job_config.get('id', 'unknown')
    job_type = job_config.get('type', 'unknown')
    
    logger.info(f"Starting job {job_id} of type {job_type}")
    
    try:
        # Record start time
        start_time = time.time()
        
        # Implement job execution based on type
        if job_type == 'process_file':
            result = process_file_job(job_config)
        elif job_type == 'generate_report':
            result = generate_report_job(job_config)
        elif job_type == 'custom':
            result = custom_job(job_config)
        else:
            raise ValueError(f"Unsupported job type: {job_type}")
        
        # Record end time
        end_time = time.time()
        
        # Create job result
        job_result = {
            'job_id': job_id,
            'status': 'success',
            'execution_time': end_time - start_time,
            'result': result
        }
        
        logger.info(f"Job {job_id} completed successfully in {job_result['execution_time']:.2f} seconds")
        
        return job_result
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        
        # Create job result with error
        job_result = {
            'job_id': job_id,
            'status': 'error',
            'error': str(e)
        }
        
        return job_result

def process_file_job(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a process file job"""
    # Extract job parameters
    input_file = config.get('input_file')
    output_file = config.get('output_file')
    
    if not input_file:
        raise ValueError("Input file not specified")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Simulate processing
    logger.info(f"Processing file: {input_file}")
    time.sleep(1)  # Simulate work
    
    # Return result
    return {
        'input_file': input_file,
        'output_file': output_file,
        'records_processed': 100  # Example metric
    }

def generate_report_job(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a report generation job"""
    # Extract job parameters
    report_type = config.get('report_type', 'unknown')
    output_file = config.get('output_file')
    
    # Simulate processing
    logger.info(f"Generating report of type: {report_type}")
    time.sleep(2)  # Simulate work
    
    # Return result
    return {
        'report_type': report_type,
        'output_file': output_file,
        'report_size': 1024  # Example metric
    }

def custom_job(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a custom job"""
    # Extract job parameters
    params = config.get('parameters', {})
    
    # Simulate processing
    logger.info(f"Running custom job with {len(params)} parameters")
    time.sleep(0.5)  # Simulate work
    