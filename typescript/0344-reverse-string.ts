/**
 * 344 - Reverse String
 * Two pointers | Middle convergance
 * Time: O(n) | Space: O(1)
 * Do not return anything, modify s in-place instead.
 */
function reverseString(s: string[]): void {
  let l = 0;
  let r = s.length - 1;

  while (l < r) {
    let left = s[l];
    let right = s[r];
    s[l] = right;
    s[r] = left;

    l++;
    r--;
  }
}
