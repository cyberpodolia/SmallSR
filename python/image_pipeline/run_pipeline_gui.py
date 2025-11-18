
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import sys
import os
from pathlib import Path
import threading

# --- Import the core processing logic ---
# We assume this GUI script is in the same directory as the original script.
try:
    from image_pipeline.run_pipeline_custom_V3_mtread import run_custom_pipeline, process_single_image
except ImportError as e:
    # Add a fallback path for local development/testing if needed
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    try:
        from image_pipeline.run_pipeline_custom_V3_mtread import run_custom_pipeline
    except ImportError:
        # If it still fails, show an error and exit.
        root = tk.Tk()
        root.withdraw() # Hide the main window
        tk.messagebox.showerror("Import Error", f"Failed to import 'run_custom_pipeline'. Make sure the script is in the correct directory.\nError: {e}")
        sys.exit(1)


class TextRedirector(object):
    """A class to redirect stdout/stderr to a Tkinter text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str, (self.tag,))
        self.widget.configure(state='disabled')
        self.widget.see(tk.END) # Auto-scroll

    def flush(self):
        pass


class PipelineGUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Image Processing Pipeline")
        self.master.geometry("800x600")
        self.pack(fill=tk.BOTH, expand=True)

        self.processing_thread = None
        self.create_widgets()
        self.redirect_logging()

    def create_widgets(self):
        # --- Frame for Controls ---
        control_frame = ttk.LabelFrame(self, text="Controls")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Input/Output directory selection
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.workers_count = tk.StringVar(value='0')
        self.start_index_name = tk.StringVar()

        ttk.Label(control_frame, text="Input Dir:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.input_dir, width=60).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(control_frame, text="Output Dir:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.output_dir, width=60).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(control_frame, text="Browse...", command=self.browse_output).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(control_frame, text="Workers:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.workers_count, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(control_frame, text="Start Index File:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.start_index_name, width=60).grid(row=3, column=1, padx=5, pady=5)

        # --- Frame for Actions ---
        action_frame = ttk.Frame(self)
        action_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing_thread)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # The exit button will use a more forceful exit to stop background threads
        self.exit_button = ttk.Button(action_frame, text="Exit", command=self.force_exit)
        self.exit_button.pack(side=tk.RIGHT, padx=5)


        # --- Frame for Log Output ---
        log_frame = ttk.LabelFrame(self, text="Log Output")
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_widget = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD, height=10)
        self.log_widget.pack(fill=tk.BOTH, expand=True)
        self.log_widget.tag_config('stdout', foreground='black')
        self.log_widget.tag_config('stderr', foreground='red')

    def browse_input(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.input_dir.set(dir_path)

    def browse_output(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir.set(dir_path)

    def redirect_logging(self):
        """Redirect print statements to the log widget."""
        sys.stdout = TextRedirector(self.log_widget, 'stdout')
        sys.stderr = TextRedirector(self.log_widget, 'stderr')

    def start_processing_thread(self):
        input_dir = self.input_dir.get()
        output_dir = self.output_dir.get()
        workers_str = self.workers_count.get()
        start_index_name = self.start_index_name.get().strip()

        if not input_dir or not os.path.isdir(input_dir):
            tk.messagebox.showerror("Error", "Please select a valid input directory.")
            return
        if not output_dir:
            tk.messagebox.showerror("Error", "Please select an output directory.")
            return

        try:
            num_workers = int(workers_str)
        except (ValueError, TypeError):
            print(f"Invalid worker count. Defaulting to 0 (auto).")
            num_workers = 0

        # Disable start button to prevent multiple runs
        self.start_button.config(state=tk.DISABLED)

        # --- Get Image Files ---
        input_path = Path(input_dir)
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        image_files = sorted([f for f in input_path.glob('*') if f.suffix.lower() in supported_extensions])

        if not image_files:
            print(f"No image files found in {input_dir}")
            self.start_button.config(state=tk.NORMAL)
            return

        # --- Determine files to process based on start index ---
        start_index = 0
        if start_index_name:
            try:
                # Find the index of the starting image
                file_names = [f.name for f in image_files]
                start_index = file_names.index(start_index_name)
                print(f"Starting processing from image: {start_index_name} (index {start_index})")
            except ValueError:
                print(f"Error: Start image '{start_index_name}' not found in the input directory.")
                tk.messagebox.showerror("Error", f"Start image '{start_index_name}' not found in the input directory.")
                self.start_button.config(state=tk.NORMAL)
                return

        files_to_process = image_files[start_index:]

        # Run the pipeline in a separate thread
        self.processing_thread = threading.Thread(
            target=self.run_pipeline_wrapper,
            args=(files_to_process, output_dir, num_workers),
            daemon=True  # Allows main thread to exit even if this thread is running
        )
        self.processing_thread.start()

    def run_pipeline_wrapper(self, image_files, output_dir, num_workers):
        """Wrapper to run the pipeline and re-enable the button."""
        try:
            print("--- Pipeline Started ---")
            run_custom_pipeline(image_paths=image_files, output_dir_path=output_dir, num_workers=num_workers)
            print("\n--- Pipeline Finished ---")
        except Exception as e:
            print(f"An error occurred in the pipeline: {e}", file=sys.stderr)
        finally:
            # Re-enable the start button when done
            self.start_button.config(state=tk.NORMAL)

    def force_exit(self):
        """
        Forcefully exits the application.
        This is a hard exit and will terminate all threads, which is what's
        needed to stop the processing if it's stuck.
        """
        print("Exit button clicked. Terminating application...")
        # os._exit(0) is a hard exit. It's not clean, but it will stop everything.
        self.master.destroy()
        os._exit(0)


if __name__ == '__main__':
    root = tk.Tk()
    app = PipelineGUI(master=root)
    # Ensure the force_exit is called when the window is closed
    root.protocol("WM_DELETE_WINDOW", app.force_exit)
    app.mainloop()
