def load_settings(filename="settings.txt"):
    """
    Load settings from a text file and return a dictionary with parsed values.
    
    Args:
        filename (str): Path to settings file
        
    Returns:
        dict: Dictionary with all settings parsed to appropriate types
    """
    settings = {}
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Split into key and value
                if '=' not in line:
                    print(f"Warning: Line {line_num} has no '=', skipping: {line}")
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if not key:
                    print(f"Warning: Line {line_num} has empty key, skipping")
                    continue
                
                # Parse value based on its format
                # First, remove quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                    settings[key] = value
                    continue
                
                # Try to parse as int
                try:
                    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        settings[key] = int(value)
                        continue
                except:
                    pass
                
                # Try to parse as float
                try:
                    float_val = float(value)
                    settings[key] = float_val
                    continue
                except:
                    pass
                
                # Try to parse as boolean
                if value.lower() == 'true':
                    settings[key] = True
                    continue
                elif value.lower() == 'false':
                    settings[key] = False
                    continue
                
                # Try to parse as tuple/list of numbers
                if ',' in value:
                    parts = [p.strip() for p in value.split(',')]
                    # Check if all parts can be converted to numbers
                    numeric_parts = []
                    for p in parts:
                        try:
                            if '.' in p:
                                numeric_parts.append(float(p))
                            else:
                                numeric_parts.append(int(p))
                        except:
                            # If any part fails, keep as string tuple
                            numeric_parts = None
                            break
                    
                    if numeric_parts is not None:
                        # Keep as tuple for coordinates/dimensions
                        settings[key] = tuple(numeric_parts)
                        continue
                    else:
                        # Keep as tuple of strings
                        settings[key] = tuple(parts)
                        continue
                
                # Default: keep as string
                settings[key] = value
    
    except FileNotFoundError:
        print(f"Settings file '{filename}' not found, using defaults")
        return {}
    except Exception as e:
        print(f"Error reading settings file: {e}")
        return {}
    
    return settings


def unpack_settings(settings_dict):
    """
    Unpack settings dictionary into individual variables with defaults.
    
    Args:
        settings_dict (dict): Settings dictionary from load_settings()
        
    Returns:
        tuple: All settings in order: target_fps, force_rgb, capture_format,
               debug_gl_finish, gl_finish_interval, overlay_size, 
               overlay_pos, radius_rgb, sigma_rgb, shader_path,
               foveal_radius, transition_width
    """
    # Define defaults
    defaults = {
        'target_fps': 60,
        'force_rgb': False,
        'capture_format': 'rgb',
        'debug_gl_finish': True,
        'gl_finish_interval': 60,
        'overlay_size': (2560, 1440),
        'overlay_pos': (0, 0),
        'radius_rgb': (0, 2, 6),
        'sigma_rgb': (0.001, 1.0, 3.0),
        'shader_path': 'shader/blur.glsl',
        'foveal_radius': 0.08,
        'transition_width': 0.12,
        'gaze_source': 'tobii',
        'blur_active': True,
        'participant_id': 0,
        'session_id': 0,
        'log_gaze': False,
        'log_path': 'gaze_log.csv',
        'lum_correction': 0.0,
    }
    
    # Update defaults with loaded settings
    for key, value in settings_dict.items():
        if key in defaults:
            # Type check for tuples
            if key in ['overlay_size', 'overlay_pos', 'radius_rgb', 'sigma_rgb']:
                if isinstance(value, tuple) and len(value) == len(defaults[key]):
                    defaults[key] = value
                else:
                    print(f"Warning: '{key}' should be a tuple of {len(defaults[key])} values, using default")
            else:
                defaults[key] = value
    
    return (
        defaults['target_fps'],
        defaults['force_rgb'],
        defaults['capture_format'],
        defaults['debug_gl_finish'],
        defaults['gl_finish_interval'],
        defaults['overlay_size'],
        defaults['overlay_pos'],
        defaults['radius_rgb'],
        defaults['sigma_rgb'],
        defaults['shader_path'],
        defaults['foveal_radius'],
        defaults['transition_width'],
        defaults['gaze_source'],
        defaults['blur_active'],
        defaults['participant_id'],
        defaults['session_id'],
        defaults['log_gaze'],
        defaults['log_path'],
        defaults['lum_correction'],
    )


#Test
if __name__ == "__main__":
    settings = load_settings("settings.txt")
    
    #Unpack
    (target_fps, force_rgb, capture_format, debug_gl_finish,
     gl_finish_interval, overlay_size, overlay_pos,
     radius_rgb, sigma_rgb, shader_path, foveal_radius, transition_width,
     gaze_source, blur_active, participant_id, session_id,
     log_gaze, log_path) = unpack_settings(settings)

    #Verify from settings.txt
    print("Loaded settings:")
    print(f"target_fps = {target_fps}")
    print(f"force_rgb = {force_rgb}")
    print(f"capture_format = {capture_format}")
    print(f"debug_gl_finish = {debug_gl_finish}")
    print(f"gl_finish_interval = {gl_finish_interval}")
    print(f"overlay_size = {overlay_size}")
    print(f"overlay_pos = {overlay_pos}")
    print(f"radius_rgb = {radius_rgb}")
    print(f"sigma_rgb = {sigma_rgb}")
    print(f"shader_path = {shader_path}")
    print(f"foveal_radius = {foveal_radius}")
    print(f"transition_width = {transition_width}")
    print(f"gaze_source = {gaze_source}")
    print(f"blur_active = {blur_active}")
    print(f"participant_id = {participant_id}")
    print(f"session_id = {session_id}")
    print(f"log_gaze = {log_gaze}")
    print(f"log_path = {log_path}")