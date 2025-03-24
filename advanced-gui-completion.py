# Return result
    return {
        'custom_job': True,
        'parameters': params
    }

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging level
    numeric_level = getattr(logging, args.log_level)
    logger.setLevel(numeric_level)
    
    try:
        # Load configuration
        config = load_configuration(args.config)
        
        # Create output directory if specified
        if args.output_dir and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Get jobs from configuration
        jobs = config.get('jobs', [])
        
        if not jobs:
            logger.warning("No jobs found in configuration")
            return 0
        
        logger.info(f"Found {len(jobs)} jobs in configuration")
        
        # Run jobs in parallel
        job_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            # Submit jobs
            future_to_job = {executor.submit(run_job, job): job for job in jobs}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                
                try:
                    result = future.result()
                    job_results.append(result)
                    
                    # Save result to output directory if specified
                    if args.output_dir:
                        job_id = job.get('id', 'unknown')
                        output_file = os.path.join(args.output_dir, f"{job_id}_result.json")
                        
                        with open(output_file, 'w') as f:
                            json.dump(result, f, indent=2)
                
                except Exception as e:
                    logger.error(f"Job execution failed: {e}")
        
        # Summarize results
        success_count = sum(1 for r in job_results if r.get('status') == 'success')
        error_count = sum(1 for r in job_results if r.get('status') == 'error')
        
        logger.info(f"Job execution complete: {success_count} succeeded, {error_count} failed")
        
        return 0 if error_count == 0 else 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
"""
        )
    ]
    
    def __init__(self):
        self.templates = self.DEFAULT_TEMPLATES.copy()
        self.custom_templates = []
    
    def get_templates(self):
        """Get all available templates"""
        return self.templates + self.custom_templates
    
    def get_template_by_name(self, name):
        """Get a template by name"""
        for template in self.get_templates():
            if template.name == name:
                return template
        return None
    
    def add_template(self, template):
        """Add a new custom template"""
        self.custom_templates.append(template)
    
    def remove_template(self, name):
        """Remove a custom template"""
        template = self.get_template_by_name(name)
        if template in self.custom_templates:
            self.custom_templates.remove(template)
            return True
        return False
    
    def create_script_from_template(self, template_name, variables, output_path):
        """Create a script from a template"""
        template = self.get_template_by_name(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Render the template
        script_content = template.render(variables)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable if needed
        if template.language in ('python', 'bash'):
            os.chmod(output_path, 0o755)
        
        return output_path

class ScriptTemplateDialog(tk.Toplevel):
    """Dialog for creating scripts from templates"""
    def __init__(self, master=None, template_manager=None, callback=None):
        super().__init__(master)
        
        self.template_manager = template_manager or ScriptTemplateManager()
        self.callback = callback
        
        # Configure dialog
        self.title("Create Script from Template")
        self.geometry("600x500")
        self.transient(master)
        self.grab_set()
        
        # Configure style
        self.configure(bg=THEME['bg'])
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets"""
        # Main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Template selection
        ttk.Label(self.main_frame, text="Select Template:", font=THEME['font_bold']).pack(anchor=tk.W, pady=(0, 5))
        
        self.template_frame = ttk.Frame(self.main_frame)
        self.template_frame.pack(fill=tk.X, pady=5)
        
        # Get templates
        templates = self.template_manager.get_templates()
        template_names = [t.name for t in templates]
        
        # Template dropdown
        self.template_var = tk.StringVar(value=template_names[0] if template_names else "")
        self.template_combo = ttk.Combobox(self.template_frame, textvariable=self.template_var, 
                                         values=template_names, state="readonly", width=40)
        self.template_combo.pack(side=tk.LEFT, padx=5)
        self.template_combo.bind("<<ComboboxSelected>>", self.on_template_changed)
        
        # Output file
        ttk.Label(self.main_frame, text="Output File:", font=THEME['font_bold']).pack(anchor=tk.W, pady=(10, 5))
        
        self.output_frame = ttk.Frame(self.main_frame)
        self.output_frame.pack(fill=tk.X, pady=5)
        
        self.output_var = tk.StringVar()
        self.output_entry = ttk.Entry(self.output_frame, textvariable=self.output_var, width=50)
        self.output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_button = ThemedButton(self.output_frame, text="Browse", command=self.browse_output)
        self.browse_button.pack(side=tk.RIGHT, padx=5)
        
        # Template variables
        ttk.Label(self.main_frame, text="Template Variables:", font=THEME['font_bold']).pack(anchor=tk.W, pady=(10, 5))
        
        # Create scrollable frame for variables
        self.vars_canvas = tk.Canvas(self.main_frame, bg=THEME['bg'], highlightthickness=0)
        self.vars_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.vars_canvas.yview)
        
        self.vars_frame = ttk.Frame(self.vars_canvas)
        self.vars_frame.bind("<Configure>", lambda e: self.vars_canvas.configure(scrollregion=self.vars_canvas.bbox("all")))
        
        self.vars_canvas.create_window((0, 0), window=self.vars_frame, anchor=tk.NW)
        self.vars_canvas.configure(yscrollcommand=self.vars_scrollbar.set)
        
        self.vars_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)
        self.vars_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Variable entries
        self.var_entries = {}
        
        # Update variables for initial template
        self.on_template_changed()
        
        # Buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.create_button = ThemedButton(self.button_frame, text="Create Script", command=self.create_script)
        self.create_button.pack(side=tk.RIGHT, padx=5)
        
        self.cancel_button = ThemedButton(self.button_frame, text="Cancel", command=self.destroy)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)
    
    def on_template_changed(self, event=None):
        """Handle template selection changes"""
        # Clear existing variable entries
        for widget in self.vars_frame.winfo_children():
            widget.destroy()
        
        self.var_entries = {}
        
        # Get selected template
        template_name = self.template_var.get()
        template = self.template_manager.get_template_by_name(template_name)
        
        if not template:
            return
        
        # Show template description
        ttk.Label(self.vars_frame, text=f"Description: {template.description}").grid(
            row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5
        )
        
        ttk.Label(self.vars_frame, text=f"Language: {template.language}").grid(
            row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5
        )
        
        # Create entries for template variables
        for i, var_name in enumerate(template.variables):
            ttk.Label(self.vars_frame, text=f"{var_name}:").grid(
                row=i+2, column=0, sticky=tk.W, padx=5, pady=5
            )
            
            var = tk.StringVar()
            
            # Set default values
            if var_name == 'author':
                import getpass
                var.set(getpass.getuser())
            elif var_name == 'date':
                import datetime
                var.set(datetime.datetime.now().strftime('%Y-%m-%d'))
            elif var_name == 'script_name':
                var.set(os.path.splitext(os.path.basename(self.output_var.get()))[0] if self.output_var.get() else '')
            
            entry = ttk.Entry(self.vars_frame, textvariable=var, width=40)
            entry.grid(row=i+2, column=1, sticky=tk.EW, padx=5, pady=5)
            
            self.var_entries[var_name] = var
        
        # Configure grid
        self.vars_frame.columnconfigure(1, weight=1)
        
        # Update canvas scroll region
        self.vars_frame.update_idletasks()
        self.vars_canvas.configure(scrollregion=self.vars_canvas.bbox("all"))
    
    def browse_output(self):
        """Browse for output file location"""
        # Get selected template for file extension
        template_name = self.template_var.get()
        template = self.template_manager.get_template_by_name(template_name)
        
        if not template:
            return
        
        # Default extension based on language
        extension = '.py'
        if template.language == 'bash':
            extension = '.sh'
        elif template.language == 'javascript':
            extension = '.js'
        
        # Get file path
        filepath = filedialog.asksaveasfilename(
            title="Save Script As",
            defaultextension=extension,
            filetypes=[
                (f"{template.language.capitalize()} files", f"*{extension}"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.output_var.set(filepath)
            
            # Update script_name variable if it exists
            if 'script_name' in self.var_entries:
                script_name = os.path.splitext(os.path.basename(filepath))[0]
                self.var_entries['script_name'].set(script_name)
    
    def create_script(self):
        """Create the script from template"""
        # Get selected template
        template_name = self.template_var.get()
        template = self.template_manager.get_template_by_name(template_name)
        
        if not template:
            messagebox.showerror("Error", "No template selected")
            return
        
        # Get output path
        output_path = self.output_var.get()
        
        if not output_path:
            messagebox.showerror("Error", "No output file specified")
            return
        
        # Collect variable values
        variables = {name: var.get() for name, var in self.var_entries.items()}
        
        try:
            # Create the script
            created_path = self.template_manager.create_script_from_template(
                template_name, variables, output_path
            )
            
            messagebox.showinfo("Success", f"Script created at {created_path}")
            
            # Call callback if provided
            if self.callback:
                self.callback(created_path)
            
            # Close dialog
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create script: {e}")

# ======================================================================
# Main Application
# ======================================================================

class SystemBuilderApp(tk.Tk):
    """Main application for the System Builder"""
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("System Builder")
        self.geometry("1200x800")
        self.minsize(800, 600)
        
        # Configure style
        setup_styles()
        
        # Initialize managers
        self.template_manager = ScriptTemplateManager()
        
        # Create widgets
        self.create_widgets()
        
        # Initial state
        self.current_workflow_path = None
        
        # Bind events
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Load configuration
        self.load_config()
    
    def create_widgets(self):
        """Create application widgets"""
        # Main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create menu
        self.create_menu()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create main content
        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas
        self.canvas_frame = ttk.Frame(self.content_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.workflow_canvas = WorkflowCanvas(self.canvas_frame, width=800, height=600)
        self.workflow_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create property panel
        self.property_frame = ttk.Frame(self.content_frame, width=300)
        self.property_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.property_frame.pack_propagate(False)
        
        self.property_panel = PropertyPanel(self.property_frame, on_property_change=self.on_property_change)
        self.property_panel.pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        self.statusbar = ttk.Frame(self, relief=tk.SUNKEN, borderwidth=1)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.statusbar, text="Ready")
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Initialize workflow controller
        self.workflow = WorkflowController(self.workflow_canvas, self.property_panel)
        
        # Add a welcome message to the canvas
        self.show_welcome_message()
    
    def create_menu(self):
        """Create application menu"""
        self.menu = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(self.menu, tearoff=0)
        file_menu.add_command(label="New Workflow", command=self.new_workflow, accelerator="Ctrl+N")
        file_menu.add_command(label="Open Workflow...", command=self.open_workflow, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_workflow, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_workflow_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export...", command=self.export_workflow)
        file_menu.add_command(label="Import...", command=self.import_workflow)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close, accelerator="Alt+F4")
        
        self.menu.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(self.menu, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Cut", command=self.cut, accelerator="Ctrl+X")
        edit_menu.add_command(label="Copy", command=self.copy, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self.paste, accelerator="Ctrl+V")
        edit_menu.add_separator()
        edit_menu.add_command(label="Delete", command=self.delete_selected, accelerator="Del")
        edit_menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        
        self.menu.add_cascade(label="Edit", menu=edit_menu)
        
        # View menu
        view_menu = tk.Menu(self.menu, tearoff=0)
        view_menu.add_command(label="Zoom In", command=lambda: self.workflow_canvas.zoom(1.1, 
                                                                                     self.workflow_canvas.winfo_width()/2,
                                                                                     self.workflow_canvas.winfo_height()/2),
                           accelerator="Ctrl++")
        view_menu.add_command(label="Zoom Out", command=lambda: self.workflow_canvas.zoom(0.9,
                                                                                      self.workflow_canvas.winfo_width()/2,
                                                                                      self.workflow_canvas.winfo_height()/2),
                           accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Zoom", command=self.workflow_canvas.reset_zoom, accelerator="Ctrl+0")
        view_menu.add_separator()
        view_menu.add_command(label="Show Grid", command=self.toggle_grid)
        
        self.menu.add_cascade(label="View", menu=view_menu)
        
        # Script menu
        script_menu = tk.Menu(self.menu, tearoff=0)
        script_menu.add_command(label="Create Script from Template...", command=self.create_script_from_template)
        script_menu.add_command(label="Add Existing Script...", command=self.add_existing_script)
        script_menu.add_separator()
        script_menu.add_command(label="Manage Templates...", command=self.manage_templates)
        
        self.menu.add_cascade(label="Script", menu=script_menu)
        
        # Workflow menu
        workflow_menu = tk.Menu(self.menu, tearoff=0)
        workflow_menu.add_command(label="Execute Workflow", command=self.execute_workflow, accelerator="F5")
        workflow_menu.add_command(label="Validate Workflow", command=self.validate_workflow)
        workflow_menu.add_separator()
        workflow_menu.add_command(label="Add Node", cascade=self.create_node_menu())
        
        self.menu.add_cascade(label="Workflow", menu=workflow_menu)
        
        # Tools menu
        tools_menu = tk.Menu(self.menu, tearoff=0)
        tools_menu.add_command(label="Settings...", command=self.show_settings)
        tools_menu.add_command(label="Log Viewer...", command=self.show_log_viewer)
        
        self.menu.add_cascade(label="Tools", menu=tools_menu)
        
        # Help menu
        help_menu = tk.Menu(self.menu, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="Examples", command=self.show_examples)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
        
        self.menu.add_cascade(label="Help", menu=help_menu)
        
        # Set the menu
        self.config(menu=self.menu)
        
        # Bind keyboard shortcuts
        self.bind("<Control-n>", lambda e: self.new_workflow())
        self.bind("<Control-o>", lambda e: self.open_workflow())
        self.bind("<Control-s>", lambda e: self.save_workflow())
        self.bind("<Control-Shift-S>", lambda e: self.save_workflow_as())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-x>", lambda e: self.cut())
        self.bind("<Control-c>", lambda e: self.copy())
        self.bind("<Control-v>", lambda e: self.paste())
        self.bind("<Delete>", lambda e: self.delete_selected())
        self.bind("<Control-a>", lambda e: self.select_all())
        self.bind("<F5>", lambda e: self.execute_workflow())
    
    def create_node_menu(self):
        """Create a menu for adding nodes"""
        node_menu = tk.Menu(self.menu, tearoff=0)
        node_menu.add_command(label="Script Node", 
                            command=lambda: self.workflow.add_node("script", 
                                                                self.workflow_canvas.winfo_width()/2,
                                                                self.workflow_canvas.winfo_height()/2))
        node_menu.add_command(label="Data Node", 
                            command=lambda: self.workflow.add_node("data", 
                                                                self.workflow_canvas.winfo_width()/2,
                                                                self.workflow_canvas.winfo_height()/2))
        node_menu.add_command(label="Computation Node", 
                            command=lambda: self.workflow.add_node("computation", 
                                                                self.workflow_canvas.winfo_width()/2,
                                                                self.workflow_canvas.winfo_height()/2))
        node_menu.add_command(label="Condition Node", 
                            command=lambda: self.workflow.add_node("condition", 
                                                                self.workflow_canvas.winfo_width()/2,
                                                                self.workflow_canvas.winfo_height()/2))
        node_menu.add_command(label="Resource Node", 
                            command=lambda: self.workflow.add_node("resource", 
                                                                self.workflow_canvas.winfo_width()/2,
                                                                self.workflow_canvas.winfo_height()/2))
        node_menu.add_command(label="Merge Node", 
                            command=lambda: self.workflow.add_node("merge", 
                                                                self.workflow_canvas.winfo_width()/2,
                                                                self.workflow_canvas.winfo_height()/2))
        return node_menu
    
    def create_toolbar(self):
        """Create application toolbar"""
        self.toolbar = ttk.Frame(self.main_container)
        self.toolbar.pack(fill=tk.X, pady=2)
        
        # New button
        self.new_button = ThemedButton(self.toolbar, text="New", command=self.new_workflow)
        self.new_button.pack(side=tk.LEFT, padx=2)
        
        # Open button
        self.open_button = ThemedButton(self.toolbar, text="Open", command=self.open_workflow)
        self.open_button.pack(side=tk.LEFT, padx=2)
        
        # Save button
        self.save_button = ThemedButton(self.toolbar, text="Save", command=self.save_workflow)
        self.save_button.pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Undo button
        self.undo_button = ThemedButton(self.toolbar, text="Undo", command=self.undo)
        self.undo_button.pack(side=tk.LEFT, padx=2)
        
        # Redo button
        self.redo_button = ThemedButton(self.toolbar, text="Redo", command=self.redo)
        self.redo_button.pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Add Script button
        self.add_script_button = ThemedButton(self.toolbar, text="Add Script", command=self.add_existing_script)
        self.add_script_button.pack(side=tk.LEFT, padx=2)
        
        # Create Script button
        self.create_script_button = ThemedButton(self.toolbar, text="Create Script", 
                                              command=self.create_script_from_template)
        self.create_script_button.pack(side=tk.LEFT, padx=2)
        
        # Separator
        ttk.Separator(self.toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        # Execute button
        self.execute_button = ThemedButton(self.toolbar, text="Execute", command=self.execute_workflow)
        self.execute_button.pack(side=tk.LEFT, padx=2)
        
        # Validate button
        self.validate_button = ThemedButton(self.toolbar, text="Validate", command=self.validate_workflow)
        self.validate_button.pack(side=tk.LEFT, padx=2)
    
    def show_welcome_message(self):
        """Show welcome message on the canvas"""
        # Clear canvas
        for node_id in list(self.workflow.nodes.keys()):
            self.workflow.delete_node(node_id)
        
        # Calculate center
        center_x = self.workflow_canvas.winfo_width() / 2
        center_y = self.workflow_canvas.winfo_height() / 2
        
        if center_x <= 1 or center_y <= 1:
            # Canvas not sized yet, use defaults
            center_x = 400
            center_y = 300
        
        # Create welcome text
        welcome_text = self.workflow_canvas.create_text(
            center_x, center_y - 50,
            text="Welcome to System Builder",
            font=("Segoe UI", 24, "bold"),
            fill=THEME['fg']
        )
        
        # Instructions
        instructions = [
            "• Create a new workflow or open an existing one",
            "• Add script nodes to the canvas",
            "• Connect nodes to define execution flow",
            "• Configure node properties",
            "• Execute the workflow"
        ]
        
        instruction_text = self.workflow_canvas.create_text(
            center_x, center_y + 30,
            text="\n".join(instructions),
            font=("Segoe UI", 12),
            fill=THEME['fg']
        )
        
        # Tag as welcome
        self.workflow_canvas.addtag_withtag("welcome", welcome_text)
        self.workflow_canvas.addtag_withtag("welcome", instruction_text)
    
    def clear_welcome_message(self):
        """Clear welcome message from canvas"""
        self.workflow_canvas.delete("welcome")
    
    def new_workflow(self):
        """Create a new workflow"""
        # Check for unsaved changes
        if self.check_unsaved_changes():
            # Clear current workflow
            self.workflow.clear_canvas()
            
            # Reset current path
            self.current_workflow_path = None
            
            # Show welcome message
            self.show_welcome_message()
            
            # Update status
            self.status_label.config(text="New workflow created")
    
    def open_workflow(self):
        """Open a workflow from file"""
        # Check for unsaved changes
        if not self.check_unsaved_changes():
            return
        
        # Show file dialog
        filepath = filedialog.askopenfilename(
            title="Open Workflow",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        # Clear welcome message
        self.clear_welcome_message()
        
        # Import the workflow
        if self.workflow.import_workflow(filepath):
            self.current_workflow_path = filepath
            self.status_label.config(text=f"Opened workflow: {filepath}")
    
    def save_workflow(self):
        """Save the current workflow"""
        if not self.current_workflow_path:
            # No current path, use save as
            return self.save_workflow_as()
        
        # Export the workflow
        if self.workflow.export_workflow():
            self.status_label.config(text=f"Saved workflow to {self.current_workflow_path}")
            return True
        
        return False
    
    def save_workflow_as(self):
        """Save the workflow to a new file"""
        # Show file dialog
        filepath = filedialog.asksaveasfilename(
            title="Save Workflow As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not filepath:
            return False
        
        # Set current path
        self.current_workflow_path = filepath
        
        # Save
        return self.save_workflow()
    
    def export_workflow(self):
        """Export the workflow"""
        self.workflow.export_workflow()
    
    def import_workflow(self):
        """Import a workflow"""
        # Check for unsaved changes
        if self.check_unsaved_changes():
            self.workflow.import_workflow()
    
    def check_unsaved_changes(self):
        """Check for unsaved changes and prompt to save"""
        # TODO: Implement change tracking
        # For now, just ask
        return True
    
    def undo(self):
        """Undo the last action"""
        # TODO: Implement undo
        self.status_label.config(text="Undo not implemented yet")
    
    def redo(self):
        """Redo the last undone action"""
        # TODO: Implement redo
        self.status_label.config(text="Redo not implemented yet")
    
    def cut(self):
        """Cut selected node"""
        # TODO: Implement cut
        self.status_label.config(text="Cut not implemented yet")
    
    def copy(self):
        """Copy selected node"""
        # TODO: Implement copy
        self.status_label.config(text="Copy not implemented yet")
    
    def paste(self):
        """Paste copied node"""
        # TODO: Implement paste
        self.status_label.config(text="Paste not implemented yet")
    
    def delete_selected(self):
        """Delete selected node"""
        node_id = self.workflow_canvas.selected_node_id
        if node_id:
            self.workflow.delete_node(node_id)
            self.status_label.config(text="Node deleted")
    
    def select_all(self):
        """Select all nodes"""
        # TODO: Implement select all
        self.status_label.config(text="Select all not implemented yet")
    
    def toggle_grid(self):
        """Toggle grid visibility"""
        # TODO: Implement grid toggle
        self.status_label.config(text="Grid toggle not implemented yet")
    
    def create_script_from_template(self):
        """Create a script from template"""
        # Create dialog
        dialog = ScriptTemplateDialog(self, self.template_manager, self.add_script_to_workflow)
    
    def add_script_to_workflow(self, script_path):
        """Add a script to the workflow"""
        # Clear welcome message
        self.clear_welcome_message()
        
        # Create a script node
        script_name = os.path.basename(script_path)
        node_type = "script"
        
        node_id = self.workflow.add_node(
            node_type,
            self.workflow_canvas.winfo_width() / 2,
            self.workflow_canvas.winfo_height() / 2,
            script_name,
            {'path': script_path, 'script_type': 'python'}  # Default to Python
        )
        
        # Update properties based on file extension
        _, ext = os.path.splitext(script_path)
        
        if ext.lower() == '.py':
            script_type = 'python'
        elif ext.lower() in ('.sh', '.bash'):
            script_type = 'shell'
        elif ext.lower() == '.js':
            script_type = 'javascript'
        else:
            script_type = 'python'  # Default
        
        # Update node properties
        node_data = self.workflow.get_node_data(node_id)
        if node_data:
            node_data['properties']['script_type'] = script_type
            self.workflow.update_node(node_id, properties=node_data['properties'])
        
        self.status_label.config(text=f"Added script: {script_name}")
    
    def add_existing_script(self):
        """Add an existing script to the workflow"""
        # Show file dialog
        filepath = filedialog.askopenfilename(
            title="Select Script",
            filetypes=[
                ("Python files", "*.py"),
                ("Shell scripts", "*.sh *.bash"),
                ("JavaScript files", "*.js"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        # Add the script
        self.add_script_to_workflow(filepath)
    
    def manage_templates(self):
        """Manage script templates"""
        # TODO: Implement template management
        self.status_label.config(text="Template management not implemented yet")
    
    def execute_workflow(self):
        """Execute the current workflow"""
        self.workflow.execute_workflow()
    
    def validate_workflow(self):
        """Validate the current workflow"""
        # TODO: Implement validation
        self.status_label.config(text="Workflow validation not implemented yet")
    
    def show_settings(self):
        """Show settings dialog"""
        # TODO: Implement settings
        self.status_label.config(text="Settings not implemented yet")
    
    def show_log_viewer(self):
        """Show log viewer"""
        # TODO: Implement log viewer
        self.status_label.config(text="Log viewer not implemented yet")
    
    def show_documentation(self):
        """Show documentation"""
        # TODO: Implement documentation
        self.status_label.config(text="Documentation not implemented yet")
    
    def show_examples(self):
        """Show example workflows"""
        # TODO: Implement examples
        self.status_label.config(text="Examples not implemented yet")
    
    def show_about(self):
        """Show about dialog"""
        # Create dialog
        dialog = tk.Toplevel(self)
        dialog.title("About System Builder")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()
        
        # Configure style
        dialog.configure(bg=THEME['bg'])
        
        # Create content
        ttk.Label(dialog, text="System Builder", font=("Segoe UI", 20, "bold")).pack(pady=(20, 10))
        ttk.Label(dialog, text="Version 1.0").pack()
        ttk.Label(dialog, text="A drag and drop system for building and executing workflows").pack(pady=10)
        
        ttk.Label(dialog, text="© 2023").pack(pady=(20, 10))
        
        # Create close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=20)
    
    def on_property_change(self, node_id, title=None, properties=None):
        """Handle property changes"""
        self.workflow.update_node(node_id, title, properties)
    
    def load_config(self):
        """Load application configuration"""
        # TODO: Implement config loading
        pass
    
    def save_config(self):
        """Save application configuration"""
        # TODO: Implement config saving
        pass
    
    def on_close(self):
        """Handle application close"""
        # Check for unsaved changes
        if self.check_unsaved_changes():
            # Save config
            self.save_config()
            
            # Destroy the window
            self.destroy()

# ======================================================================
# Entry Point
# ======================================================================

def main():
    """Main entry point"""
    app = SystemBuilderApp()
    app.mainloop()

if __name__ == "__main__":
    main()
